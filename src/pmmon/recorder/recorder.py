"""Live order-book recorder and journal replay.

Pipeline::

    Polymarket WS ─┐                       ┌─> JournalWriter (raw, append-only)
    Kalshi REST poll ┼─> asyncio.Queue ──> Dispatcher ─┤
    REST resync ────┘                       └─> Normaliser ─> ParquetSink (idempotent)

A gap flagged by a normaliser triggers a REST snapshot for that market, which is itself
journaled and dispatched, so a replay of the journal reproduces resyncs exactly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import websockets

from pmmon.models import BookRow
from pmmon.recorder.journal import JournalWriter, read_journal
from pmmon.recorder.normalise import (
    KalshiRestNormaliser,
    KalshiWsNormaliser,
    PolymarketNormaliser,
)
from pmmon.recorder.sink import ParquetSink
from pmmon.venues.kalshi import KalshiClient
from pmmon.venues.polymarket import WS_MARKET_URL, PolymarketClient

log = logging.getLogger(__name__)


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class Dispatcher:
    """Routes raw records to normalisers and the sink. Pure / synchronous -> testable."""

    sink: ParquetSink
    journal: JournalWriter | None = None
    pm: PolymarketNormaliser = field(default_factory=PolymarketNormaliser)
    kalshi_rest: KalshiRestNormaliser = field(default_factory=KalshiRestNormaliser)
    kalshi_ws: KalshiWsNormaliser = field(default_factory=KalshiWsNormaliser)
    rows_out: int = 0

    def dispatch(self, venue: str, recv_ts_ms: int, kind: str, payload: Any) -> list[BookRow]:
        if self.journal is not None:
            self.journal.write(venue, recv_ts_ms, kind, payload)
        rows: list[BookRow] = []
        if venue == "polymarket":
            if kind == "ws":
                rows = self.pm.on_ws(payload, recv_ts_ms)
            elif kind == "rest_book":
                rows = self.pm.on_book(payload, recv_ts_ms, source="rest")
        elif venue == "kalshi":
            if kind == "rest_book":
                rows = self.kalshi_rest.on_rest_book(payload["ticker"], payload, recv_ts_ms)
            elif kind == "kalshi_ws":
                rows = self.kalshi_ws.on_ws(payload, recv_ts_ms)
        if rows:
            self.sink.write(rows)
            self.rows_out += len(rows)
        return rows

    def pending_resyncs(self) -> dict[str, set[str]]:
        return {
            "polymarket": set(self.pm.needs_resync),
            "kalshi": set(self.kalshi_rest.needs_resync) | set(self.kalshi_ws.needs_resync),
        }

    def stats(self) -> dict[str, Any]:
        return {
            "polymarket": vars(self.pm.stats)
            | {"gap_markets": len(set(self.pm.stats.gap_markets))},
            "kalshi_rest": vars(self.kalshi_rest.stats)
            | {"gap_markets": len(set(self.kalshi_rest.stats.gap_markets))},
            "rows_written": self.sink.written_rows,
            "duplicate_rows_skipped": self.sink.duplicate_rows,
        }


def ingest_journal(path: str | Path, sink: ParquetSink) -> Dispatcher:
    """Rebuild the Parquet store from a raw journal (idempotent)."""
    d = Dispatcher(sink=sink)
    for rec in read_journal(path):
        d.dispatch(rec.venue, rec.recv_ts_ms, rec.kind, rec.payload)
    sink.close()
    return d


class LiveRecorder:
    def __init__(
        self,
        *,
        out_dir: str | Path,
        pm_tokens: list[str],
        kalshi_tickers: list[str],
        journal_path: str | Path | None = None,
        kalshi_poll_s: float = 2.0,
        pm_client: PolymarketClient | None = None,
        kalshi_client: KalshiClient | None = None,
    ) -> None:
        self.sink = ParquetSink(out_dir, flush_rows=20_000)
        journal = JournalWriter(journal_path) if journal_path else None
        self.dispatcher = Dispatcher(sink=self.sink, journal=journal)
        self.pm_tokens = pm_tokens
        self.kalshi_tickers = kalshi_tickers
        self.kalshi_poll_s = kalshi_poll_s
        self.pm_client = pm_client or PolymarketClient()
        self.kalshi_client = kalshi_client or KalshiClient()
        self.queue: asyncio.Queue[tuple[str, int, str, Any]] = asyncio.Queue()
        self.reconnects = 0
        self._tasks: set[asyncio.Task[None]] = set()

    async def _pm_ws(self, stop: asyncio.Event) -> None:
        backoff = 1.0
        while not stop.is_set() and self.pm_tokens:
            try:
                async with websockets.connect(
                    WS_MARKET_URL, max_size=None, ping_interval=None
                ) as ws:
                    await ws.send(json.dumps({"assets_ids": self.pm_tokens, "type": "market"}))
                    backoff = 1.0
                    pinger = asyncio.create_task(self._pinger(ws))
                    try:
                        while not stop.is_set():
                            try:
                                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            except TimeoutError:
                                continue
                            if msg == "PONG":
                                continue
                            await self.queue.put(("polymarket", now_ms(), "ws", json.loads(msg)))
                    finally:
                        pinger.cancel()
            except (OSError, websockets.WebSocketException) as exc:
                self.reconnects += 1
                log.warning("polymarket ws error %s; reconnect in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    @staticmethod
    async def _pinger(ws: Any) -> None:
        while True:
            await asyncio.sleep(10)
            await ws.send("PING")

    async def _kalshi_poll(self, stop: asyncio.Event) -> None:
        while not stop.is_set() and self.kalshi_tickers:
            t0 = time.monotonic()
            for t in self.kalshi_tickers:
                if stop.is_set():
                    return
                try:
                    ob = await self.kalshi_client.orderbook(t)
                except Exception as exc:  # keep polling other markets
                    log.warning("kalshi orderbook %s failed: %s", t, exc)
                    continue
                await self.queue.put(("kalshi", now_ms(), "rest_book", {"ticker": t, **ob}))
            await asyncio.sleep(max(0.0, self.kalshi_poll_s - (time.monotonic() - t0)))

    async def _resync(self, venue: str, market: str) -> None:
        try:
            if venue == "polymarket":
                book = await self.pm_client.book(market)
                await self.queue.put(("polymarket", now_ms(), "rest_book", book))
            else:
                ob = await self.kalshi_client.orderbook(market)
                await self.queue.put(("kalshi", now_ms(), "rest_book", {"ticker": market, **ob}))
        except Exception as exc:
            log.warning("resync %s/%s failed: %s", venue, market, exc)

    async def _consume(self, stop: asyncio.Event) -> None:
        inflight: set[tuple[str, str]] = set()
        while not (stop.is_set() and self.queue.empty()):
            try:
                venue, ts, kind, payload = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            except TimeoutError:
                continue
            self.dispatcher.dispatch(venue, ts, kind, payload)
            if kind == "rest_book":
                key = (venue, payload.get("ticker") or payload.get("asset_id"))
                inflight.discard(key)
            for v, markets in self.dispatcher.pending_resyncs().items():
                for m in markets:
                    if (v, m) not in inflight and not stop.is_set():
                        inflight.add((v, m))
                        task = asyncio.create_task(self._resync(v, m))
                        self._tasks.add(task)
                        task.add_done_callback(self._tasks.discard)

    async def run(self, duration_s: float) -> dict[str, Any]:
        stop = asyncio.Event()
        tasks = [
            asyncio.create_task(self._pm_ws(stop)),
            asyncio.create_task(self._kalshi_poll(stop)),
        ]
        consumer = asyncio.create_task(self._consume(stop))
        await asyncio.sleep(duration_s)
        stop.set()
        for t in tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        await consumer
        self.sink.close()
        if self.dispatcher.journal:
            self.dispatcher.journal.close()
        await self.pm_client.aclose()
        await self.kalshi_client.aclose()
        return self.dispatcher.stats() | {"ws_reconnects": self.reconnects}
