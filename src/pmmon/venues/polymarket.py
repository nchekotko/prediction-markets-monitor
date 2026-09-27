"""Polymarket read-only adapter.

* Gamma API (``gamma-api.polymarket.com``): market / event metadata, resolution state.
* CLOB REST (``clob.polymarket.com``): order-book snapshots, price history.
* CLOB WS (``wss://ws-subscriptions-clob.polymarket.com/ws/market``): public market channel.

No authenticated / trading endpoints are used.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx

from pmmon.fees import polymarket_fee_model
from pmmon.http import RestClient
from pmmon.models import MarketStatus, UnifiedMarket, Venue

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
WS_MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def parse_ts(value: Any) -> datetime | None:
    """Parse ISO-ish timestamps used by Gamma (``2026-09-27T17:00:00Z``,
    ``2026-09-09 14:25:04+00``) and epoch milliseconds."""
    if value in (None, ""):
        return None
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value) / 1000, tz=UTC)
    s = str(value).strip().replace(" ", "T", 1)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if len(s) >= 3 and s[-3] in "+-" and s[-6] not in "+-":  # '+00' -> '+00:00'
        s = s + ":00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _json_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    try:
        out = json.loads(value)
    except (TypeError, ValueError):
        return []
    return out if isinstance(out, list) else []


def _f(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def market_status(m: dict[str, Any]) -> tuple[MarketStatus, str]:
    uma = (m.get("umaResolutionStatus") or "").lower()
    history = [str(x).lower() for x in _json_list(m.get("umaResolutionStatuses"))]
    raw = f"active={m.get('active')};closed={m.get('closed')};uma={uma or '-'}"
    if history:
        raw += ";uma_history=" + ",".join(history)
    if m.get("archived"):
        return MarketStatus.RESOLVED if uma == "resolved" else MarketStatus.CLOSED, raw
    if uma == "disputed":
        return MarketStatus.DISPUTED, raw
    if m.get("closed"):
        if uma == "resolved":
            return MarketStatus.RESOLVED, raw
        if uma == "proposed":
            return MarketStatus.PROPOSED, raw
        return MarketStatus.CLOSED, raw
    if uma == "proposed":
        return MarketStatus.PROPOSED, raw
    if not m.get("active", False):
        return MarketStatus.UNOPENED, raw
    if m.get("acceptingOrders") is False:
        return MarketStatus.PAUSED, raw
    return MarketStatus.OPEN, raw


def parse_gamma_market(
    m: dict[str, Any], event: dict[str, Any] | None = None
) -> list[UnifiedMarket]:
    """One Gamma market (a condition) -> one UnifiedMarket per outcome token."""
    outcomes = [str(o) for o in _json_list(m.get("outcomes"))]
    tokens = [str(t) for t in _json_list(m.get("clobTokenIds"))]
    prices = [_f(p) for p in _json_list(m.get("outcomePrices"))]
    if not tokens or len(tokens) != len(outcomes):
        return []
    ev = event or ((m.get("events") or [None])[0]) or {}
    status, raw = market_status(m)
    fees = polymarket_fee_model(
        m.get("feeSchedule"), m.get("category") or ev.get("category"), m.get("feesEnabled")
    )
    tick = _f(m.get("orderPriceMinTickSize")) or 0.01
    best_bid = _f(m.get("bestBid"))
    best_ask = _f(m.get("bestAsk"))
    source = m.get("resolutionSource") or ev.get("resolutionSource") or ""
    out: list[UnifiedMarket] = []
    for i, (tok, outcome) in enumerate(zip(tokens, outcomes, strict=True)):
        # Gamma reports bestBid/bestAsk for the first outcome; mirror them for the second
        if i == 0:
            bb, ba = best_bid, best_ask
        else:
            bb = None if best_ask is None else round(1 - best_ask, 6)
            ba = None if best_bid is None else round(1 - best_bid, 6)
        price = prices[i] if i < len(prices) else None
        result = price if status is MarketStatus.RESOLVED else None
        out.append(
            UnifiedMarket(
                venue=Venue.POLYMARKET,
                market_id=tok,
                parent_id=str(m.get("conditionId") or m.get("id")),
                event_id=str(ev.get("id") or m.get("id")),
                event_title=str(ev.get("title") or m.get("question") or ""),
                question=str(m.get("question") or ""),
                outcome=outcome,
                status=status,
                best_bid=bb if bb is not None and 0 <= bb <= 1 else None,
                best_ask=ba if ba is not None and 0 <= ba <= 1 else None,
                last_price=price if status is MarketStatus.OPEN else None,
                tick_size=tick,
                min_order_size=_f(m.get("orderMinSize")),
                fees=fees,
                open_time=parse_ts(m.get("startDate")),
                close_time=parse_ts(m.get("endDate")),
                expected_resolution_time=parse_ts(m.get("endDate")),
                resolution_source=str(source),
                rules_text=str(m.get("description") or ""),
                result=result,
                volume=_f(m.get("volumeNum") or m.get("volume")),
                category=m.get("category") or ev.get("category"),
                url=f"https://polymarket.com/event/{ev.get('slug') or m.get('slug')}",
                raw_status=raw,
            )
        )
    return out


def parse_book_levels(levels: list[dict[str, Any]]) -> list[tuple[float, float]]:
    return [(float(lv["price"]), float(lv["size"])) for lv in levels or []]


class PolymarketClient:
    def __init__(
        self,
        *,
        gamma_url: str = GAMMA_URL,
        clob_url: str = CLOB_URL,
        rate_per_sec: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.gamma = RestClient(gamma_url, rate_per_sec=rate_per_sec, transport=transport)
        self.clob = RestClient(clob_url, rate_per_sec=rate_per_sec, transport=transport)

    async def aclose(self) -> None:
        await self.gamma.aclose()
        await self.clob.aclose()

    async def __aenter__(self) -> PolymarketClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def iter_markets(
        self, *, page_size: int = 100, max_pages: int | None = None, **params: Any
    ) -> AsyncIterator[dict[str, Any]]:
        """Offset-paginated ``GET /markets``; stops on a short page."""
        offset = 0
        pages = 0
        while True:
            q = {"limit": page_size, "offset": offset, **params}
            batch = await self.gamma.get_json("/markets", q)
            for m in batch:
                yield m
            pages += 1
            if len(batch) < page_size or (max_pages is not None and pages >= max_pages):
                return
            offset += page_size

    async def events(self, **params: Any) -> list[dict[str, Any]]:
        return await self.gamma.get_json("/events", params)

    async def event_by_slug(self, slug: str) -> dict[str, Any] | None:
        res = await self.gamma.get_json("/events", {"slug": slug})
        return res[0] if res else None

    async def book(self, token_id: str) -> dict[str, Any]:
        return await self.clob.get_json("/book", {"token_id": token_id})

    async def price_history(
        self,
        token_id: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
        fidelity_min: int = 60,
    ) -> list[tuple[int, float]]:
        params: dict[str, Any] = {"market": token_id, "fidelity": fidelity_min}
        if start_ts is None:
            params["interval"] = "max"
        else:
            params["startTs"], params["endTs"] = start_ts, end_ts
        res = await self.clob.get_json("/prices-history", params)
        return [(int(h["t"]), float(h["p"])) for h in res.get("history", [])]
