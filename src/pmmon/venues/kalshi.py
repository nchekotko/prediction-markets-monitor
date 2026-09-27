"""Kalshi read-only adapter (public Trade API v2 market-data endpoints).

Kalshi order books list resting *bids* for YES and for NO. A NO bid at price q is
economically a YES ask at 1 - q, which is how we build a YES-side L2 book.

The Kalshi WebSocket requires API-key auth even for market data, so the default recorder
path for Kalshi is REST polling of ``/markets/{ticker}/orderbook`` (diffed into deltas).
A parser for the authenticated WS ``orderbook_snapshot``/``orderbook_delta`` messages
(with ``seq`` gap detection) lives in :mod:`pmmon.recorder.normalise` for users with keys.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from pmmon.fees import kalshi_fee_model
from pmmon.http import RestClient
from pmmon.models import FeeModel, MarketStatus, UnifiedMarket, Venue
from pmmon.venues.polymarket import parse_ts

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

_STATUS = {
    "initialized": MarketStatus.UNOPENED,
    "unopened": MarketStatus.UNOPENED,
    "active": MarketStatus.OPEN,
    "open": MarketStatus.OPEN,
    "inactive": MarketStatus.PAUSED,
    "paused": MarketStatus.PAUSED,
    "closed": MarketStatus.CLOSED,
    "determined": MarketStatus.PROPOSED,
    "amended": MarketStatus.PROPOSED,
    "disputed": MarketStatus.DISPUTED,
    "finalized": MarketStatus.RESOLVED,
    "settled": MarketStatus.RESOLVED,
}


def _f(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _dollars(m: dict[str, Any], name: str) -> float | None:
    """Read ``<name>_dollars`` (current API) falling back to legacy integer cents."""
    v = _f(m.get(f"{name}_dollars"))
    if v is None:
        cents = _f(m.get(name))
        v = None if cents is None else cents / 100
    return v


def market_result(m: dict[str, Any]) -> float | None:
    res = (m.get("result") or "").lower()
    if res == "yes":
        return 1.0
    if res == "no":
        return 0.0
    if res == "scalar":
        return _dollars(m, "settlement_value")
    return None


def tick_size(m: dict[str, Any]) -> float:
    ranges = m.get("price_ranges") or []
    steps = [_f(r.get("step")) for r in ranges]
    steps = [s for s in steps if s]
    if steps:
        return min(steps)
    legacy = _f(m.get("tick_size"))
    return legacy / 100 if legacy else 0.01


def settlement_text(*objs: dict[str, Any] | None) -> str:
    for o in objs:
        if not o:
            continue
        srcs = o.get("settlement_sources") or []
        if srcs:
            return "; ".join(f"{s.get('name', '')} ({s.get('url', '')})".strip() for s in srcs)
    return ""


def parse_market(
    m: dict[str, Any],
    event: dict[str, Any] | None = None,
    series: dict[str, Any] | None = None,
) -> UnifiedMarket:
    raw_status = (m.get("status") or "").lower()
    status = _STATUS.get(raw_status, MarketStatus.CLOSED)
    result = market_result(m)
    if (
        status is MarketStatus.RESOLVED
        and result is None
        and (m.get("result") or "") in ("", "void")
    ):
        status = MarketStatus.VOID
    fees: FeeModel = kalshi_fee_model(
        (series or {}).get("fee_type"), (series or {}).get("fee_multiplier")
    )
    yes_bid = _dollars(m, "yes_bid")
    yes_ask = _dollars(m, "yes_ask")
    event_title = (event or {}).get("title") or m.get("event_ticker", "")
    rules = "\n".join(x for x in (m.get("rules_primary"), m.get("rules_secondary")) if x)
    return UnifiedMarket(
        venue=Venue.KALSHI,
        market_id=m["ticker"],
        parent_id=m["ticker"],
        event_id=m.get("event_ticker", ""),
        event_title=event_title,
        question=m.get("title") or "",
        outcome=m.get("yes_sub_title") or m.get("subtitle") or m.get("title") or "",
        status=status,
        best_bid=yes_bid if yes_bid else None,  # 0 means "no bid"
        best_ask=yes_ask if yes_ask and yes_ask < 1 else None,
        last_price=_dollars(m, "last_price") or None,
        tick_size=tick_size(m),
        fees=fees,
        open_time=parse_ts(m.get("open_time")),
        close_time=parse_ts(m.get("close_time")),
        expected_resolution_time=parse_ts(m.get("expected_expiration_time"))
        or parse_ts(m.get("expiration_time")),
        resolution_source=settlement_text(event, series),
        rules_text=rules,
        result=result if status is MarketStatus.RESOLVED else None,
        volume=_f(m.get("volume_fp") or m.get("volume")),
        category=(event or {}).get("category") or (series or {}).get("category"),
        url=f"https://kalshi.com/markets/{m.get('event_ticker', '').lower()}",
        raw_status=raw_status,
    )


def parse_orderbook(
    payload: dict[str, Any],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Kalshi ``/orderbook`` -> YES-side (bids, asks) in probability units.

    Supports the current fixed-point format (``orderbook_fp.yes_dollars``) and the
    legacy integer-cents format (``orderbook.yes``).
    """
    if "orderbook_fp" in payload:
        ob = payload["orderbook_fp"] or {}
        yes = [(float(p), float(q)) for p, q in ob.get("yes_dollars") or []]
        no = [(float(p), float(q)) for p, q in ob.get("no_dollars") or []]
    else:
        ob = payload.get("orderbook") or {}
        yes = [(p / 100, float(q)) for p, q in ob.get("yes") or []]
        no = [(p / 100, float(q)) for p, q in ob.get("no") or []]
    bids = [(round(p, 6), q) for p, q in yes if q > 0]
    asks = [(round(1 - p, 6), q) for p, q in no if q > 0]
    return bids, asks


class KalshiClient:
    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        rate_per_sec: float = 8.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.rest = RestClient(base_url, rate_per_sec=rate_per_sec, transport=transport)
        self._series_cache: dict[str, dict[str, Any]] = {}
        self._event_cache: dict[str, dict[str, Any]] = {}

    async def aclose(self) -> None:
        await self.rest.aclose()

    async def __aenter__(self) -> KalshiClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def iter_markets(
        self, *, page_size: int = 200, max_pages: int | None = None, **params: Any
    ) -> AsyncIterator[dict[str, Any]]:
        """Cursor-paginated ``GET /markets``."""
        cursor: str | None = None
        pages = 0
        while True:
            q = {"limit": page_size, "cursor": cursor, **params}
            res = await self.rest.get_json("/markets", q)
            for m in res.get("markets", []):
                yield m
            pages += 1
            cursor = res.get("cursor") or None
            if not cursor or (max_pages is not None and pages >= max_pages):
                return

    async def iter_events(
        self, *, page_size: int = 200, max_pages: int | None = None, **params: Any
    ) -> AsyncIterator[dict[str, Any]]:
        cursor: str | None = None
        pages = 0
        while True:
            res = await self.rest.get_json(
                "/events", {"limit": page_size, "cursor": cursor, **params}
            )
            for e in res.get("events", []):
                yield e
            pages += 1
            cursor = res.get("cursor") or None
            if not cursor or (max_pages is not None and pages >= max_pages):
                return

    async def event(self, event_ticker: str) -> dict[str, Any]:
        if event_ticker not in self._event_cache:
            res = await self.rest.get_json(f"/events/{event_ticker}")
            self._event_cache[event_ticker] = res.get("event", {})
        return self._event_cache[event_ticker]

    async def series(self, series_ticker: str) -> dict[str, Any]:
        if series_ticker not in self._series_cache:
            res = await self.rest.get_json(f"/series/{series_ticker}")
            self._series_cache[series_ticker] = res.get("series", {})
        return self._series_cache[series_ticker]

    async def unified(self, m: dict[str, Any]) -> UnifiedMarket:
        """Enrich a raw market with its event/series (cached) and normalise it."""
        ev = await self.event(m["event_ticker"]) if m.get("event_ticker") else None
        ser = await self.series(ev["series_ticker"]) if ev and ev.get("series_ticker") else None
        return parse_market(m, ev, ser)

    async def orderbook(self, ticker: str, depth: int | None = None) -> dict[str, Any]:
        return await self.rest.get_json(f"/markets/{ticker}/orderbook", {"depth": depth})

    async def candlesticks(
        self, series_ticker: str, ticker: str, start_ts: int, end_ts: int, period_min: int = 60
    ) -> list[dict[str, Any]]:
        res = await self.rest.get_json(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_min},
        )
        return res.get("candlesticks", [])
