"""Reference strategies for the event-time backtester."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pmmon.backtest.engine import Context, Fill, Order, Strategy
from pmmon.models import Side
from pmmon.monitor.metrics import arb_for_size


def _floor_tick(p: float, tick: float) -> float:
    return round(math.floor(p / tick + 1e-9) * tick, 6)


def _ceil_tick(p: float, tick: float) -> float:
    return round(math.ceil(p / tick - 1e-9) * tick, 6)


@dataclass
class PassiveMarketMaker(Strategy):
    """Quote both sides around the microprice with a linear inventory skew.

    reservation = microprice - skew * inventory / max_inventory
    bid = min(floor(reservation - half_spread), best_bid)   (never improve through the touch
    ask = max(ceil(reservation + half_spread), best_ask)     unless ``improve`` is set)

    Quotes are post-only, re-quoted when the target price changes (throttled by
    ``min_requote_ms``) and the side that would breach ``max_inventory`` is pulled.
    """

    markets: list[str]
    tick: dict[str, float] = field(default_factory=dict)
    size: float = 100.0
    half_spread: float = 0.005
    max_inventory: float = 1_000.0
    skew: float = 0.02
    min_requote_ms: int = 500
    min_book_spread: float = 0.0
    improve: bool = False
    _last_quote: dict[str, int] = field(default_factory=dict)

    def on_book(self, ctx: Context, market: str) -> None:
        if market not in self.markets:
            return
        if ctx.now_ms - self._last_quote.get(market, -(10**12)) < self.min_requote_ms:
            return
        book = ctx.book(market)
        bb, ba, micro = book.best_bid, book.best_ask, book.microprice
        if bb is None or ba is None or micro is None or book.is_crossed:
            self._cancel_all(ctx, market)
            return
        tick = self.tick.get(market, 0.01)
        inv = ctx.position(market)
        res = micro - self.skew * inv / self.max_inventory
        bid = _floor_tick(res - self.half_spread, tick)
        ask = _ceil_tick(res + self.half_spread, tick)
        if not self.improve:
            bid, ask = min(bid, bb), max(ask, ba)
        bid = min(bid, _floor_tick(ba - tick, tick))
        ask = max(ask, _ceil_tick(bb + tick, tick))
        want = {
            Side.BID: bid
            if (
                inv + self.size <= self.max_inventory
                and bid >= tick
                and ba - bb >= self.min_book_spread
            )
            else None,
            Side.ASK: ask
            if (
                inv - self.size >= -self.max_inventory
                and ask <= 1 - tick
                and ba - bb >= self.min_book_spread
            )
            else None,
        }
        for side, px in want.items():
            live = [o for o in ctx.open_orders(market, tag="mm") if o.side is side]
            keep = [o for o in live if px is not None and abs(o.price - px) < 1e-9]
            for o in live:
                if o not in keep:
                    ctx.cancel(o)
            if px is not None and not keep:
                ctx.submit(market, side, px, self.size, "post", tag="mm")
        self._last_quote[market] = ctx.now_ms

    @staticmethod
    def _cancel_all(ctx: Context, market: str) -> None:
        for o in ctx.open_orders(market, tag="mm"):
            ctx.cancel(o)


@dataclass
class ArbPair:
    a: str
    b: str
    risk_flags: list[str] = field(default_factory=list)


@dataclass
class CrossVenueArb(Strategy):
    """Take both legs when the book-swept edge net of fees exceeds ``min_edge``.

    Signals use the book *as seen now*; the IOC legs reach each venue after that venue's
    latency and execute against the book at arrival - so latency erodes or kills the edge
    and can leave one leg unfilled (leg risk, valued at mid in the report).
    """

    pairs: list[ArbPair]
    size: float = 100.0
    min_edge: float = 0.005  # per contract, after fees
    slippage_per_contract: float = 0.0
    cooldown_ms: int = 2_000
    max_position: float = 2_000.0
    signals: int = 0
    attempts: int = 0
    legs: dict[int, list[Order]] = field(default_factory=dict)
    _last: dict[tuple[str, str], int] = field(default_factory=dict)
    _by_market: dict[str, list[ArbPair]] = field(default_factory=dict)
    signal_log: list[tuple[int, str, str, float, float]] = field(default_factory=list)

    def on_start(self, ctx: Context) -> None:
        for p in self.pairs:
            self._by_market.setdefault(p.a, []).append(p)
            self._by_market.setdefault(p.b, []).append(p)

    def on_book(self, ctx: Context, market: str) -> None:
        for p in self._by_market.get(market, []):
            if p.a not in ctx.markets or p.b not in ctx.markets:
                continue
            for buy, sell in ((p.a, p.b), (p.b, p.a)):
                self._check(ctx, buy, sell)

    def _check(self, ctx: Context, buy: str, sell: str) -> None:
        sa, sb = ctx.markets[buy], ctx.markets[sell]
        if sa.book.best_ask is None or sb.book.best_bid is None:
            return
        if sb.book.best_bid <= sa.book.best_ask:
            return  # cheap pre-check: no crossed quotes across venues
        q = arb_for_size(sa.book, sa.fees, sb.book, sb.fees, self.size, self.slippage_per_contract)
        if q.size <= 0 or q.net_edge < self.min_edge:
            return
        self.signals += 1
        self.signal_log.append((ctx.now_ms, buy, sell, q.net_edge, q.size))
        key = (buy, sell)
        if ctx.now_ms - self._last.get(key, -(10**12)) < self.cooldown_ms:
            return
        if abs(ctx.position(buy)) + q.size > self.max_position:
            return
        self._last[key] = ctx.now_ms
        self.attempts += 1
        # limit prices = worst level used by the sweep -> no worse than the signal
        buy_lim = _worst_level(sa.book.sorted_levels(Side.ASK), q.size)
        sell_lim = _worst_level(sb.book.sorted_levels(Side.BID), q.size)
        if buy_lim is None or sell_lim is None:
            return
        o1 = ctx.submit(buy, Side.BID, buy_lim, q.size, "ioc", tag="arb")
        o2 = ctx.submit(sell, Side.ASK, sell_lim, q.size, "ioc", tag="arb")
        self.legs[self.attempts] = [o1, o2]

    def leg_stats(self, fills: list[Fill]) -> dict[str, float]:
        """Leg outcomes and P&L *locked in* by hedged quantity (valid only if both venues
        resolve the same way - see mapping risk flags)."""
        by_oid: dict[int, list[Fill]] = {}
        for f in fills:
            by_oid.setdefault(f.oid, []).append(f)
        full = partial = none = 0
        unhedged = 0.0
        locked = 0.0
        for o1, o2 in self.legs.values():
            f1, f2 = o1.filled, o2.filled
            hedged = min(f1, f2)
            if hedged > 0:
                b = by_oid.get(o1.oid, [])
                s_ = by_oid.get(o2.oid, [])
                avg_b = sum(x.price * x.qty for x in b) / f1
                avg_s = sum(x.price * x.qty for x in s_) / f2
                fees = sum(x.fee for x in b) * hedged / f1 + sum(x.fee for x in s_) * hedged / f2
                locked += hedged * (avg_s - avg_b) - fees
            if f1 >= o1.size - 1e-9 and f2 >= o2.size - 1e-9:
                full += 1
            elif f1 <= 1e-9 and f2 <= 1e-9:
                none += 1
            else:
                partial += 1
            unhedged += abs(f1 - f2)
        return {
            "signals": self.signals,
            "attempts": self.attempts,
            "both_legs_filled": full,
            "partial_or_one_leg": partial,
            "no_fill": none,
            "unhedged_contracts": round(unhedged, 2),
            "locked_pnl_if_same_resolution_usd": round(locked, 4),
        }


def _worst_level(levels: list[tuple[float, float]], size: float) -> float | None:
    got = 0.0
    for p, s in levels:
        got += s
        if got >= size - 1e-9:
            return p
    return levels[-1][0] if levels else None
