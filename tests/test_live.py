"""Smoke tests against the real public APIs. Deselected by default; run with ``-m live``."""

from __future__ import annotations

import pytest

from pmmon.models import MarketStatus
from pmmon.venues.kalshi import KalshiClient, parse_orderbook
from pmmon.venues.polymarket import PolymarketClient, parse_gamma_market

pytestmark = pytest.mark.live


async def test_polymarket_gamma_and_clob_live() -> None:
    async with PolymarketClient() as pm:
        markets = [
            m
            async for m in pm.iter_markets(page_size=5, max_pages=1, active="true", closed="false")
        ]
        contracts = [c for m in markets for c in parse_gamma_market(m)]
        assert contracts and all(c.status is MarketStatus.OPEN for c in contracts)
        book = await pm.book(contracts[0].market_id)
        assert "bids" in book and "asks" in book


async def test_kalshi_markets_and_orderbook_live() -> None:
    async with KalshiClient() as k:
        mk = [
            m
            async for m in k.iter_markets(
                page_size=5, max_pages=1, status="open", mve_filter="exclude"
            )
        ]
        assert mk
        u = await k.unified(mk[0])
        assert u.market_id == mk[0]["ticker"]
        bids, asks = parse_orderbook(await k.orderbook(mk[0]["ticker"]))
        assert all(0 <= p <= 1 for p, _ in bids + asks)
