"""Venue message -> normalised :class:`BookRow` stream, with gap detection.

Each normaliser keeps an :class:`L2Book` per market so that it can

* emit *absolute* level sizes for deltas (the persisted format is "set level to size"),
* detect sequence gaps / state divergence and ask the recorder to resync from REST,
* emit periodic synthetic snapshots so any time slice of the Parquet store is replayable
  without reading from the beginning of time.

Gap detection per venue:

* Polymarket market channel has no sequence numbers. Every ``price_change`` carries the
  venue's ``best_bid``/``best_ask`` *after* the change; if our reconstructed top of book
  disagrees, a message was lost or reordered -> gap. Exchange timestamps going backwards are
  counted as well.
* Kalshi WS (authenticated) carries a per-subscription ``seq``; ``seq != last + 1`` -> gap.
* Kalshi REST polling cannot lose messages; consecutive polls are diffed into deltas.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pmmon.book import L2Book
from pmmon.models import BookEventKind, BookRow, Side, Venue
from pmmon.venues.kalshi import parse_orderbook

DEFAULT_SNAPSHOT_INTERVAL_MS = 5 * 60 * 1000
PRICE_EPS = 1e-9


@dataclass
class GapStats:
    gaps: int = 0
    resyncs: int = 0
    out_of_order: int = 0
    dropped_before_snapshot: int = 0
    messages: int = 0
    rows: int = 0
    gap_markets: list[str] = field(default_factory=list)


class _BaseNormaliser:
    venue: Venue

    def __init__(self, snapshot_interval_ms: int = DEFAULT_SNAPSHOT_INTERVAL_MS) -> None:
        self.books: dict[str, L2Book] = {}
        self.local_seq: dict[str, int] = {}
        self.last_snapshot_ts: dict[str, int] = {}
        self.needs_resync: set[str] = set()
        self.snapshot_interval_ms = snapshot_interval_ms
        self.stats = GapStats()

    def _next_seq(self, market: str) -> int:
        s = self.local_seq.get(market, 0) + 1
        self.local_seq[market] = s
        return s

    def _flag_gap(self, market: str) -> None:
        self.stats.gaps += 1
        self.stats.gap_markets.append(market)
        self.needs_resync.add(market)

    def _snapshot_rows(
        self, market: str, book: L2Book, ts: int, recv: int, seq: int, event_id: str
    ) -> list[BookRow]:
        rows = [
            BookRow(
                self.venue, market, ts, recv, seq, BookEventKind.SNAPSHOT, Side.BID, p, s, event_id
            )
            for p, s in book.sorted_levels(Side.BID)
        ]
        rows += [
            BookRow(
                self.venue, market, ts, recv, seq, BookEventKind.SNAPSHOT, Side.ASK, p, s, event_id
            )
            for p, s in book.sorted_levels(Side.ASK)
        ]
        if not rows:  # empty book: keep a marker row so replay knows the book was cleared
            rows.append(
                BookRow(
                    self.venue,
                    market,
                    ts,
                    recv,
                    seq,
                    BookEventKind.SNAPSHOT,
                    Side.BID,
                    0.0,
                    0.0,
                    event_id,
                )
            )
        self.last_snapshot_ts[market] = ts
        return rows

    def _apply_snapshot(
        self,
        market: str,
        bids: Iterable[tuple[float, float]],
        asks: Iterable[tuple[float, float]],
        ts: int,
        recv: int,
        event_id: str,
        seq: int | None = None,
    ) -> list[BookRow]:
        book = self.books.setdefault(market, L2Book(market))
        book.reset(bids, asks, ts)
        book.last_update_recv_ms = recv
        if market in self.needs_resync:
            self.needs_resync.discard(market)
            self.stats.resyncs += 1
        s = seq if seq is not None else self._next_seq(market)
        book.seq = s
        return self._snapshot_rows(market, book, ts, recv, s, event_id)

    def _maybe_synthetic_snapshot(self, market: str, ts: int, recv: int) -> list[BookRow]:
        last = self.last_snapshot_ts.get(market)
        if last is None or ts - last < self.snapshot_interval_ms or market in self.needs_resync:
            return []
        book = self.books[market]
        return self._snapshot_rows(market, book, ts, recv, book.seq, f"synth:{ts}")

    def _count(self, rows: list[BookRow]) -> list[BookRow]:
        self.stats.messages += 1
        self.stats.rows += len(rows)
        return rows


class PolymarketNormaliser(_BaseNormaliser):
    venue = Venue.POLYMARKET

    def on_ws(self, payload: Any, recv_ts_ms: int) -> list[BookRow]:
        if isinstance(payload, list):
            out: list[BookRow] = []
            for p in payload:
                out += self.on_ws(p, recv_ts_ms)
            return out
        if not isinstance(payload, dict):
            return []
        et = payload.get("event_type")
        if et == "book":
            return self._count(self.on_book(payload, recv_ts_ms, source="book"))
        if et == "price_change":
            return self._count(self._on_price_change(payload, recv_ts_ms))
        if et == "last_trade_price":
            return self._count(self._on_trade(payload, recv_ts_ms))
        return []

    def on_book(self, p: dict[str, Any], recv_ts_ms: int, source: str = "rest") -> list[BookRow]:
        market = str(p["asset_id"])
        ts = int(p.get("timestamp") or recv_ts_ms)
        bids = [(float(x["price"]), float(x["size"])) for x in p.get("bids") or []]
        asks = [(float(x["price"]), float(x["size"])) for x in p.get("asks") or []]
        return self._apply_snapshot(
            market, bids, asks, ts, recv_ts_ms, f"{source}:{ts}:{p.get('hash', '')}"
        )

    def _on_price_change(self, p: dict[str, Any], recv: int) -> list[BookRow]:
        ts = int(p.get("timestamp") or recv)
        changes = p.get("price_changes")
        if changes is None:  # legacy format: one asset, list under "changes"
            changes = [{**c, "asset_id": p.get("asset_id")} for c in p.get("changes") or []]
        rows: list[BookRow] = []
        reported: dict[str, tuple[float | None, float | None]] = {}
        touched: list[str] = []
        for ch in changes:
            market = str(ch["asset_id"])
            book = self.books.get(market)
            if book is None:
                self.stats.dropped_before_snapshot += 1
                self.needs_resync.add(market)
                continue
            if market not in touched:
                touched.append(market)
                if ts < book.ts_ms:
                    self.stats.out_of_order += 1
                book.seq = self._next_seq(market)
            side = Side.BID if str(ch["side"]).upper() == "BUY" else Side.ASK
            price, size = float(ch["price"]), float(ch["size"])
            book.set_level(side, price, size)
            book.ts_ms = max(book.ts_ms, ts)
            book.last_update_recv_ms = recv
            rows.append(
                BookRow(
                    self.venue,
                    market,
                    ts,
                    recv,
                    book.seq,
                    BookEventKind.DELTA,
                    side,
                    price,
                    size,
                    f"pc:{ts}:{ch.get('hash', '')}",
                )
            )
            if "best_bid" in ch or "best_ask" in ch:
                reported[market] = (_opt(ch.get("best_bid")), _opt(ch.get("best_ask")))
        for market in touched:
            book = self.books[market]
            if market in reported and market not in self.needs_resync:
                bb, ba = reported[market]
                if not (_same(bb, book.best_bid) and _same(ba, book.best_ask)):
                    self._flag_gap(market)
            rows += self._maybe_synthetic_snapshot(market, ts, recv)
        return rows

    def _on_trade(self, p: dict[str, Any], recv: int) -> list[BookRow]:
        market = str(p["asset_id"])
        ts = int(p.get("timestamp") or recv)
        # taker BUY lifts a resting ask; taker SELL hits a resting bid
        side = Side.ASK if str(p.get("side", "")).upper() == "BUY" else Side.BID
        seq = self.books[market].seq if market in self.books else 0
        return [
            BookRow(
                self.venue,
                market,
                ts,
                recv,
                seq,
                BookEventKind.TRADE,
                side,
                float(p["price"]),
                float(p["size"]),
                f"tr:{ts}:{p.get('transaction_hash', '')}",
            )
        ]


class KalshiRestNormaliser(_BaseNormaliser):
    """Diffs consecutive REST order-book polls into deltas."""

    venue = Venue.KALSHI

    def on_rest_book(self, ticker: str, payload: dict[str, Any], recv_ts_ms: int) -> list[BookRow]:
        bids, asks = parse_orderbook(payload)
        prev = self.books.get(ticker)
        if prev is None or ticker in self.needs_resync:
            return self._count(
                self._apply_snapshot(
                    ticker, bids, asks, recv_ts_ms, recv_ts_ms, f"poll:{recv_ts_ms}"
                )
            )
        new = L2Book(ticker)
        new.reset(bids, asks, recv_ts_ms)
        seq = self._next_seq(ticker)
        rows: list[BookRow] = []
        for side, old_lv, new_lv in (
            (Side.BID, prev.bids, new.bids),
            (Side.ASK, prev.asks, new.asks),
        ):
            for price in sorted(set(old_lv) | set(new_lv)):
                o, n = old_lv.get(price, 0.0), new_lv.get(price, 0.0)
                if abs(o - n) > PRICE_EPS:
                    rows.append(
                        BookRow(
                            self.venue,
                            ticker,
                            recv_ts_ms,
                            recv_ts_ms,
                            seq,
                            BookEventKind.DELTA,
                            side,
                            price,
                            n,
                            f"poll:{recv_ts_ms}",
                        )
                    )
                    prev.set_level(side, price, n)
        prev.ts_ms = recv_ts_ms
        prev.last_update_recv_ms = recv_ts_ms
        prev.seq = seq
        rows += self._maybe_synthetic_snapshot(ticker, recv_ts_ms, recv_ts_ms)
        return self._count(rows)


class KalshiWsNormaliser(_BaseNormaliser):
    """Authenticated Kalshi WS ``orderbook_snapshot`` / ``orderbook_delta`` messages."""

    venue = Venue.KALSHI

    def __init__(self, snapshot_interval_ms: int = DEFAULT_SNAPSHOT_INTERVAL_MS) -> None:
        super().__init__(snapshot_interval_ms)
        self.last_seq: dict[int, int] = {}  # per subscription id
        self.sid_markets: dict[int, set[str]] = {}

    def on_ws(self, payload: dict[str, Any], recv_ts_ms: int) -> list[BookRow]:
        typ = payload.get("type")
        if typ not in ("orderbook_snapshot", "orderbook_delta"):
            return []
        sid = int(payload.get("sid", 0))
        seq = int(payload.get("seq", 0))
        msg = payload.get("msg") or {}
        ticker = msg["market_ticker"]
        self.sid_markets.setdefault(sid, set()).add(ticker)
        last = self.last_seq.get(sid)
        self.last_seq[sid] = seq
        ts = int(msg.get("ts_ms") or recv_ts_ms)
        if typ == "orderbook_snapshot":
            yes = [(float(p), float(q)) for p, q in msg.get("yes_dollars_fp") or []]
            no = [(float(p), float(q)) for p, q in msg.get("no_dollars_fp") or []]
            bids = [(round(p, 6), q) for p, q in yes]
            asks = [(round(1 - p, 6), q) for p, q in no]
            return self._count(
                self._apply_snapshot(ticker, bids, asks, ts, recv_ts_ms, f"ws:{sid}:{seq}", seq=seq)
            )
        if last is not None and seq != last + 1:
            for m in self.sid_markets.get(sid, {ticker}):
                self._flag_gap(m)
        book = self.books.get(ticker)
        if book is None or ticker in self.needs_resync:
            self.stats.dropped_before_snapshot += 1
            self.needs_resync.add(ticker)
            return []
        price = float(msg["price_dollars"])
        delta = float(msg["delta_fp"])
        if msg.get("side") == "yes":
            side, level = Side.BID, round(price, 6)
            cur = book.bids.get(level, 0.0)
        else:
            side, level = Side.ASK, round(1 - price, 6)
            cur = book.asks.get(level, 0.0)
        new_size = max(0.0, cur + delta)
        book.set_level(side, level, new_size)
        book.seq, book.ts_ms, book.last_update_recv_ms = seq, ts, recv_ts_ms
        rows = [
            BookRow(
                self.venue,
                ticker,
                ts,
                recv_ts_ms,
                seq,
                BookEventKind.DELTA,
                side,
                level,
                new_size,
                f"ws:{sid}:{seq}",
            )
        ]
        rows += self._maybe_synthetic_snapshot(ticker, ts, recv_ts_ms)
        return self._count(rows)


def _opt(v: Any) -> float | None:
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def _same(a: float | None, b: float | None) -> bool:
    """Compare venue-reported vs reconstructed touch; venue reports 0/"" for an empty side."""
    a = None if a is None or a == 0.0 else a
    b = None if b is None or b == 0.0 else b
    if a is None or b is None:
        return a is b
    return abs(a - b) < 1e-9
