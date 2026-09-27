"""Event-time backtester over recorded L2 books.

Replays persisted :class:`BookRow` events strictly in exchange-time order across venues.
Strategies act through :class:`Context`; every order pays a per-venue latency before it
reaches the (replayed) book.

Fill models
-----------
* **IOC / taker**: executes at arrival time by sweeping the replayed book up to a limit
  price. The replayed book is not modified (no market impact) - size is capped at the depth
  that was actually displayed, which is the main source of optimism; see README.
* **Passive / maker (queue-position model)**: a new order joins the back of the queue, so
  ``queue_ahead`` = displayed size at that price on arrival. Then
    - a trade print at our price and side consumes ``queue_ahead`` first, the excess fills us;
    - a level decrease not explained by trade prints is split into cancellations
      (reduce ``queue_ahead`` pro rata: cancels are assumed uniformly spread over the queue)
      and - for venues without public trade prints (Kalshi via REST polling) - hidden
      trades, a ``hidden_trade_share`` fraction of the decrease;
    - if the opposite touch trades through our price, we are filled in full at our price.

Accounting identity (checked in tests)::

    total_pnl = cash + sum(position * final_mid)
              = spread_capture + inventory_pnl - fees

where spread capture is measured against the mid at fill time and inventory P&L is
position times subsequent mid changes.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from pmmon.book import L2Book
from pmmon.fees import fee as fee_fn
from pmmon.models import BookEventKind, BookRow, FeeModel, Side, Venue
from pmmon.recorder.sink import iter_events

OrderType = Literal["ioc", "post"]


@dataclass
class Order:
    oid: int
    market: str
    side: Side  # BID = buy YES, ASK = sell YES
    price: float
    size: float
    otype: OrderType
    submit_ts: int
    arrive_ts: int
    tag: str = ""
    filled: float = 0.0
    queue_ahead: float = 0.0
    active: bool = False
    done: bool = False
    cancel_at: int | None = None
    seen_levels: dict[float, float] | None = None  # IOC on a polled venue: liquidity at arrival

    @property
    def remaining(self) -> float:
        return self.size - self.filled


@dataclass(frozen=True)
class Fill:
    ts: int
    oid: int
    market: str
    side: Side
    price: float
    qty: float
    fee: float
    maker: bool
    mid_at_fill: float | None
    tag: str


@dataclass
class MarketState:
    venue: Venue
    fees: FeeModel
    book: L2Book
    position: float = 0.0
    last_mid: float | None = None
    mid_ts: list[int] = field(default_factory=list)
    mid_val: list[float] = field(default_factory=list)


@dataclass
class BacktestConfig:
    latency_ms: dict[Venue, int] = field(
        default_factory=lambda: {Venue.POLYMARKET: 250, Venue.KALSHI: 150}
    )
    hidden_trade_share: dict[Venue, float] = field(
        default_factory=lambda: {Venue.POLYMARKET: 0.0, Venue.KALSHI: 0.5}
    )
    markout_horizons_ms: tuple[int, ...] = (1_000, 10_000, 60_000)
    # Venues recorded by REST polling only show the book every few seconds. An IOC there
    # is filled only against liquidity present both at arrival and ``confirm_ms`` later
    # (roughly one poll interval) - otherwise stale snapshots produce phantom fills.
    ioc_confirm_ms: dict[Venue, int] = field(
        default_factory=lambda: {Venue.POLYMARKET: 0, Venue.KALSHI: 3_000}
    )


class Strategy:
    def on_start(self, ctx: Context) -> None: ...

    def on_book(self, ctx: Context, market: str) -> None: ...

    def on_trade(
        self, ctx: Context, market: str, side: Side, price: float, size: float
    ) -> None: ...

    def on_fill(self, ctx: Context, fill: Fill) -> None: ...

    def on_end(self, ctx: Context) -> None: ...


class Context:
    def __init__(self, markets: dict[str, MarketState], cfg: BacktestConfig) -> None:
        self.markets = markets
        self.cfg = cfg
        self.now_ms = 0
        self.cash = 0.0
        self.fees_paid = 0.0
        self.spread_capture = 0.0
        self.inventory_pnl = 0.0
        self.fills: list[Fill] = []
        # only live (not done) orders, per market -> per-event work is O(live orders)
        self.live: dict[str, dict[int, Order]] = defaultdict(dict)
        self.cancelling: list[Order] = []
        self.n_orders = 0
        self._ids = itertools.count(1)
        self.pending: list[Order] = []  # not yet arrived, sorted by arrive_ts
        self.quotes_placed = 0
        self.inventory_path: list[tuple[int, str, float]] = []

    # ---- strategy API -----------------------------------------------------------------
    def book(self, market: str) -> L2Book:
        return self.markets[market].book

    def position(self, market: str) -> float:
        return self.markets[market].position

    def submit(
        self, market: str, side: Side, price: float, size: float, otype: OrderType, tag: str = ""
    ) -> Order:
        st = self.markets[market]
        lat = self.cfg.latency_ms.get(st.venue, 0)
        o = Order(
            next(self._ids),
            market,
            side,
            round(price, 6),
            size,
            otype,
            self.now_ms,
            self.now_ms + lat,
            tag,
        )
        self.live[market][o.oid] = o
        self.n_orders += 1
        bisect.insort(self.pending, o, key=lambda x: (x.arrive_ts, x.oid))
        if otype == "post":
            self.quotes_placed += 1
        return o

    def cancel(self, order: Order) -> None:
        if order.done:
            return
        lat = self.cfg.latency_ms.get(self.markets[order.market].venue, 0)
        order.cancel_at = self.now_ms + lat
        if not order.active:  # still in flight: cancel on arrival
            self._finish(order)
        else:
            self.cancelling.append(order)

    def _finish(self, order: Order) -> None:
        order.done = True
        order.active = False
        self.live[order.market].pop(order.oid, None)

    def open_orders(self, market: str, tag: str | None = None) -> list[Order]:
        return [o for o in self.live[market].values() if tag is None or o.tag == tag]

    # ---- engine internals -----------------------------------------------------------------
    def _record_fill(self, o: Order, price: float, qty: float, maker: bool) -> Fill:
        st = self.markets[o.market]
        f = fee_fn(st.fees, price, qty, maker=maker)
        sign = 1 if o.side is Side.BID else -1
        st.position += sign * qty
        self.cash -= sign * price * qty + f
        self.fees_paid += f
        # Reference mid = the last *marked* mid, so that MTM changes split exactly into
        # spread capture (at fill) + inventory P&L (at subsequent marks).
        ref = st.last_mid
        if ref is None:
            bm = st.book.mid
            ref = bm if bm is not None and not st.book.is_crossed else price
            st.last_mid = ref
        self.spread_capture += sign * (ref - price) * qty
        mid = ref
        o.filled += qty
        if o.remaining <= 1e-9:
            self._finish(o)
        fill = Fill(self.now_ms, o.oid, o.market, o.side, price, qty, f, maker, mid, o.tag)
        self.fills.append(fill)
        self.inventory_path.append((self.now_ms, o.market, st.position))
        return fill


class Backtester:
    def __init__(
        self,
        rows: list[BookRow],
        fees: dict[str, FeeModel],
        strategy: Strategy,
        cfg: BacktestConfig | None = None,
    ) -> None:
        self.rows = rows
        self.cfg = cfg or BacktestConfig()
        venues = {r.market_id: r.venue for r in rows}
        self.markets = {
            m: MarketState(v, fees.get(m, FeeModel()), L2Book(m)) for m, v in venues.items()
        }
        self.ctx = Context(self.markets, self.cfg)
        self.strategy = strategy
        self.trade_prints: dict[str, float] = defaultdict(float)  # per (market, price, side)

    # ---- order arrival / matching -------------------------------------------------------
    def _arrive(self, o: Order) -> None:
        ctx = self.ctx
        ctx.now_ms = max(ctx.now_ms, o.arrive_ts)
        if o.done:
            return
        book = self.markets[o.market].book
        if o.otype == "ioc":
            take_side = Side.ASK if o.side is Side.BID else Side.BID
            levels = book.sorted_levels(take_side)
            confirm = self.cfg.ioc_confirm_ms.get(self.markets[o.market].venue, 0)
            if confirm > 0 and o.seen_levels is None:
                o.seen_levels = dict(levels)
                o.arrive_ts += confirm
                bisect.insort(self.ctx.pending, o, key=lambda x: (x.arrive_ts, x.oid))
                return
            if o.seen_levels is not None:
                seen = o.seen_levels
                levels = [(p, min(q, seen[p])) for p, q in levels if p in seen]
            fills: list[tuple[float, float]] = []
            need = o.size
            for price, avail in levels:
                if (o.side is Side.BID and price > o.price + 1e-9) or (
                    o.side is Side.ASK and price < o.price - 1e-9
                ):
                    break
                q = min(avail, need)
                fills.append((price, q))
                need -= q
                if need <= 1e-9:
                    break
            for price, q in fills:
                f = ctx._record_fill(o, price, q, maker=False)
                self.strategy.on_fill(ctx, f)
            ctx._finish(o)
            return
        # post-only: reject if it would cross
        bb, ba = book.best_bid, book.best_ask
        if (o.side is Side.BID and ba is not None and o.price >= ba - 1e-9) or (
            o.side is Side.ASK and bb is not None and o.price <= bb + 1e-9
        ):
            ctx._finish(o)
            return
        levels = book.bids if o.side is Side.BID else book.asks
        o.queue_ahead = levels.get(o.price, 0.0)
        o.active = True

    def _process_pending(self, until_ts: int) -> None:
        p = self.ctx.pending
        while p and p[0].arrive_ts <= until_ts:
            o = p.pop(0)  # pending holds only in-flight orders: short list
            self._arrive(o)
        if self.ctx.cancelling:
            keep = []
            for o in self.ctx.cancelling:
                if o.done:
                    continue
                if o.cancel_at is not None and o.cancel_at <= until_ts:
                    self.ctx._finish(o)
                else:
                    keep.append(o)
            self.ctx.cancelling = keep

    def _passive_level_change(
        self, market: str, side: Side, price: float, old: float, new: float, traded: float
    ) -> None:
        """Update queue positions of our resting orders at (side, price)."""
        venue = self.markets[market].venue
        decrease = max(0.0, old - new)
        unexplained = max(0.0, decrease - traded)
        hidden_trades = unexplained * self.cfg.hidden_trade_share.get(venue, 0.0)
        cancels = unexplained - hidden_trades
        for o in list(self.ctx.live[market].values()):
            if not o.active or o.done or o.side is not side:
                continue
            if abs(o.price - price) > 1e-9:
                continue
            # hidden trades consume the queue front first
            vol = hidden_trades
            fill_q = max(0.0, vol - o.queue_ahead)
            o.queue_ahead = max(0.0, o.queue_ahead - vol)
            if old > 0 and cancels > 0:
                o.queue_ahead = max(0.0, o.queue_ahead - cancels * (o.queue_ahead / old))
            o.queue_ahead = min(o.queue_ahead, new)
            if fill_q > 0:
                self._maker_fill(o, min(fill_q, o.remaining))

    def _on_trade(self, market: str, side: Side, price: float, size: float) -> None:
        for o in list(self.ctx.live[market].values()):
            if not o.active or o.done or o.side is not side:
                continue
            if abs(o.price - price) > 1e-9:
                # a print strictly through our price means our level was taken
                through = (side is Side.BID and price < o.price - 1e-9) or (
                    side is Side.ASK and price > o.price + 1e-9
                )
                if through:
                    self._maker_fill(o, o.remaining)
                continue
            fill_q = max(0.0, size - o.queue_ahead)
            o.queue_ahead = max(0.0, o.queue_ahead - size)
            if fill_q > 0:
                self._maker_fill(o, min(fill_q, o.remaining))

    def _cross_check(self, market: str) -> None:
        book = self.markets[market].book
        bb, ba = book.best_bid, book.best_ask
        for o in list(self.ctx.live[market].values()):
            if not o.active or o.done:
                continue
            if (o.side is Side.BID and ba is not None and ba < o.price - 1e-9) or (
                o.side is Side.ASK and bb is not None and bb > o.price + 1e-9
            ):
                self._maker_fill(o, o.remaining)

    def _maker_fill(self, o: Order, qty: float) -> None:
        if qty <= 1e-12 or o.done:
            return
        f = self.ctx._record_fill(o, o.price, qty, maker=True)
        self.strategy.on_fill(self.ctx, f)

    # ---- main loop ------------------------------------------------------------------------
    def _mark(self, st: MarketState, ts: int) -> None:
        mid = st.book.mid
        if mid is None or st.book.is_crossed:
            return
        if st.last_mid is not None and st.position != 0:
            self.ctx.inventory_pnl += st.position * (mid - st.last_mid)
        st.last_mid = mid
        if not st.mid_ts or st.mid_val[-1] != mid:
            st.mid_ts.append(ts)
            st.mid_val.append(mid)

    def run(self) -> BacktestResult:
        ctx = self.ctx
        self.strategy.on_start(ctx)
        pending_trades: dict[tuple[str, Side, float], float] = defaultdict(float)
        for ev in iter_events(self.rows):
            head = ev[0]
            self._process_pending(head.ts_ms)
            ctx.now_ms = max(ctx.now_ms, head.ts_ms)
            st = self.markets[head.market_id]
            book = st.book
            if head.kind is BookEventKind.SNAPSHOT:
                bids = [(r.price, r.size) for r in ev if r.side is Side.BID and r.size > 0]
                asks = [(r.price, r.size) for r in ev if r.side is Side.ASK and r.size > 0]
                old_b, old_a = dict(book.bids), dict(book.asks)
                book.reset(bids, asks, head.ts_ms)
                for side, old_lv, new_lv in (
                    (Side.BID, old_b, book.bids),
                    (Side.ASK, old_a, book.asks),
                ):
                    for price in set(old_lv) | set(new_lv):
                        o, n = old_lv.get(price, 0.0), new_lv.get(price, 0.0)
                        if n < o:
                            traded = pending_trades.pop((head.market_id, side, price), 0.0)
                            self._passive_level_change(head.market_id, side, price, o, n, traded)
            elif head.kind is BookEventKind.DELTA:
                for r in ev:
                    levels = book.bids if r.side is Side.BID else book.asks
                    old = levels.get(round(r.price, 6), 0.0)
                    book.set_level(r.side, r.price, r.size)
                    if r.size < old:
                        traded = pending_trades.pop((r.market_id, r.side, round(r.price, 6)), 0.0)
                        self._passive_level_change(
                            r.market_id, r.side, round(r.price, 6), old, r.size, traded
                        )
            else:  # trade prints
                for r in ev:
                    self._on_trade(r.market_id, r.side, round(r.price, 6), r.size)
                    pending_trades[(r.market_id, r.side, round(r.price, 6))] += r.size
                    self.strategy.on_trade(ctx, r.market_id, r.side, r.price, r.size)
            book.last_update_recv_ms = head.recv_ts_ms
            if head.kind is not BookEventKind.TRADE:
                self._cross_check(head.market_id)
                self._mark(st, head.ts_ms)
                self.strategy.on_book(ctx, head.market_id)
        self._process_pending(math.inf)  # type: ignore[arg-type]
        self.strategy.on_end(ctx)
        return BacktestResult.build(ctx, self.markets, self.cfg)


@dataclass
class BacktestResult:
    total_pnl: float
    spread_capture: float
    inventory_pnl: float
    fees: float
    n_fills: int
    maker_fills: int
    taker_fills: int
    quotes_placed: int
    fill_rate: float
    volume: float
    avg_spread_captured: float  # per contract, probability units
    markouts: dict[int, float]  # horizon ms -> mean markout per contract (+ = favourable)
    adverse_selection: dict[int, float]  # horizon ms -> spread captured minus markout
    max_abs_inventory: float
    final_positions: dict[str, float]
    fills: list[Fill]

    @classmethod
    def build(
        cls, ctx: Context, markets: dict[str, MarketState], cfg: BacktestConfig
    ) -> BacktestResult:
        mtm = ctx.cash + sum(
            st.position * (st.last_mid if st.last_mid is not None else 0.0)
            for st in markets.values()
        )
        vol = sum(f.qty for f in ctx.fills)
        maker = [f for f in ctx.fills if f.maker]
        markouts: dict[int, float] = {}
        adverse: dict[int, float] = {}
        edge0 = sum(
            (1 if f.side is Side.BID else -1) * (f.mid_at_fill - f.price) * f.qty
            for f in ctx.fills
            if f.mid_at_fill is not None
        )
        vol_mid = sum(f.qty for f in ctx.fills if f.mid_at_fill is not None)
        for h in cfg.markout_horizons_ms:
            tot = 0.0
            q = 0.0
            for f in ctx.fills:
                st = markets[f.market]
                m = _mid_at(st, f.ts + h)
                if m is None:
                    continue
                sign = 1 if f.side is Side.BID else -1
                tot += sign * (m - f.price) * f.qty
                q += f.qty
            markouts[h] = tot / q if q else float("nan")
            adverse[h] = (edge0 / vol_mid - markouts[h]) if vol_mid and q else float("nan")
        max_inv = max((abs(p) for _, _, p in ctx.inventory_path), default=0.0)
        return cls(
            total_pnl=mtm,
            spread_capture=ctx.spread_capture,
            inventory_pnl=ctx.inventory_pnl,
            fees=ctx.fees_paid,
            n_fills=len(ctx.fills),
            maker_fills=len(maker),
            taker_fills=len(ctx.fills) - len(maker),
            quotes_placed=ctx.quotes_placed,
            fill_rate=(len({f.oid for f in maker}) / ctx.quotes_placed)
            if ctx.quotes_placed
            else 0.0,
            volume=vol,
            avg_spread_captured=edge0 / vol_mid if vol_mid else float("nan"),
            markouts=markouts,
            adverse_selection=adverse,
            max_abs_inventory=max_inv,
            final_positions={m: st.position for m, st in markets.items() if st.position},
            fills=ctx.fills,
        )

    def summary(self) -> dict[str, object]:
        return {
            "total_pnl_usd": round(self.total_pnl, 4),
            "attribution": {
                "spread_capture_usd": round(self.spread_capture, 4),
                "inventory_pnl_usd": round(self.inventory_pnl, 4),
                "fees_usd": round(-self.fees, 4),
                "residual_usd": round(
                    self.total_pnl - (self.spread_capture + self.inventory_pnl - self.fees), 6
                ),
            },
            "fills": self.n_fills,
            "maker_fills": self.maker_fills,
            "taker_fills": self.taker_fills,
            "quotes_placed": self.quotes_placed,
            "quote_fill_rate": round(self.fill_rate, 4),
            "volume_contracts": round(self.volume, 2),
            "avg_spread_captured_cents": round(self.avg_spread_captured * 100, 3),
            "markout_cents": {f"{h // 1000}s": round(v * 100, 3) for h, v in self.markouts.items()},
            "adverse_selection_cents": {
                f"{h // 1000}s": round(v * 100, 3) for h, v in self.adverse_selection.items()
            },
            "max_abs_inventory": round(self.max_abs_inventory, 2),
            "final_positions": {k[:20]: round(v, 2) for k, v in self.final_positions.items()},
        }


def _mid_at(st: MarketState, ts: int) -> float | None:
    i = bisect.bisect_right(st.mid_ts, ts) - 1
    return st.mid_val[i] if i >= 0 else None
