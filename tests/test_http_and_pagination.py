from __future__ import annotations

import httpx
import pytest
from conftest import load

from pmmon.http import ApiError, RestClient, TokenBucket
from pmmon.venues.kalshi import KalshiClient
from pmmon.venues.polymarket import PolymarketClient


async def _no_sleep(_: float) -> None:
    return None


async def test_retries_on_429_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "0.01"}, text="slow down")
        return httpx.Response(200, json={"ok": True})

    c = RestClient(
        "https://x.test", transport=httpx.MockTransport(handler), sleep=_no_sleep, rate_per_sec=1000
    )
    assert await c.get_json("/a") == {"ok": True}
    assert calls["n"] == 3 and c.retry_count == 2
    await c.aclose()


async def test_non_retryable_status_raises() -> None:
    c = RestClient(
        "https://x.test",
        transport=httpx.MockTransport(lambda r: httpx.Response(404, text="nope")),
        sleep=_no_sleep,
    )
    with pytest.raises(ApiError) as ei:
        await c.get_json("/missing")
    assert ei.value.status == 404
    await c.aclose()


async def test_gives_up_after_max_retries() -> None:
    c = RestClient(
        "https://x.test",
        transport=httpx.MockTransport(lambda r: httpx.Response(503)),
        sleep=_no_sleep,
        max_retries=2,
        rate_per_sec=1000,
    )
    with pytest.raises(ApiError):
        await c.get_json("/down")
    assert c.request_count == 3
    await c.aclose()


async def test_token_bucket_limits_rate() -> None:
    t = {"now": 0.0}
    bucket = TokenBucket(rate=2.0, capacity=2.0, clock=lambda: t["now"])
    await bucket.acquire()
    await bucket.acquire()
    assert bucket._tokens < 1  # burst exhausted
    t["now"] += 0.5  # half a second -> one token refilled
    await bucket.acquire()
    assert bucket._tokens == pytest.approx(0.0, abs=1e-9)


async def test_kalshi_cursor_pagination() -> None:
    pages = {
        None: {"markets": [{"ticker": "A"}, {"ticker": "B"}], "cursor": "c1"},
        "c1": {"markets": [{"ticker": "C"}], "cursor": ""},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=pages[request.url.params.get("cursor")])

    async with KalshiClient(transport=httpx.MockTransport(handler), rate_per_sec=1000) as k:
        tickers = [m["ticker"] async for m in k.iter_markets(page_size=2)]
    assert tickers == ["A", "B", "C"]


async def test_polymarket_offset_pagination_stops_on_short_page() -> None:
    data = [{"id": str(i)} for i in range(5)]
    seen_offsets: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        off = int(request.url.params["offset"])
        lim = int(request.url.params["limit"])
        seen_offsets.append(off)
        return httpx.Response(200, json=data[off : off + lim])

    async with PolymarketClient(transport=httpx.MockTransport(handler), rate_per_sec=1000) as pm:
        ids = [m["id"] async for m in pm.iter_markets(page_size=2)]
    assert ids == ["0", "1", "2", "3", "4"]
    assert seen_offsets == [0, 2, 4]


async def test_kalshi_unified_enrichment_uses_cache() -> None:
    markets = load("kalshi_markets_kcmia.json")
    event = load("kalshi_event_kcmia.json")
    series = load("kalshi_series_nflgame.json")
    hits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.path)
        if request.url.path.endswith("/events/KXNFLGAME-26SEP27KCMIA"):
            return httpx.Response(200, json=event)
        if request.url.path.endswith("/series/KXNFLGAME"):
            return httpx.Response(200, json=series)
        return httpx.Response(404)

    async with KalshiClient(transport=httpx.MockTransport(handler), rate_per_sec=1000) as k:
        out = [await k.unified(m) for m in markets["markets"]]
    assert {u.outcome for u in out} == {"Kansas City", "Miami"}
    assert len(hits) == 2  # event + series fetched once each
