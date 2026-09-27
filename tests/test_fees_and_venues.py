from __future__ import annotations

import pytest
from conftest import load

from pmmon.fees import fee, kalshi_fee_model, polymarket_fee_model
from pmmon.models import FeeKind, MarketStatus, Venue
from pmmon.venues.kalshi import parse_market, parse_orderbook
from pmmon.venues.polymarket import market_status, parse_gamma_market, parse_ts

# ------------------------------------------------------------------------------ fees


def test_kalshi_taker_fee_rounds_up_to_cent() -> None:
    m = kalshi_fee_model("quadratic")
    # 0.07 * 100 * 0.5 * 0.5 = 1.75 exactly
    assert fee(m, 0.5, 100) == pytest.approx(1.75)
    # 0.07 * 1 * 0.5 * 0.5 = 0.0175 -> 0.02
    assert fee(m, 0.5, 1) == pytest.approx(0.02)
    assert fee(m, 0.5, 100, maker=True) == 0.0


def test_kalshi_maker_fees_only_for_maker_series() -> None:
    m = kalshi_fee_model("quadratic_with_maker_fees", 1)
    assert m.maker_rate == pytest.approx(0.0175)
    # 0.0175 * 100 * 0.3 * 0.7 = 0.3675 -> 0.37
    assert fee(m, 0.3, 100, maker=True) == pytest.approx(0.37)


def test_polymarket_fee_from_schedule_and_symmetry() -> None:
    m = polymarket_fee_model({"exponent": 1, "rate": 0.05, "takerOnly": True})
    assert m.kind is FeeKind.POLYMARKET
    assert fee(m, 0.2, 100) == pytest.approx(100 * 0.05 * 0.2 * 0.8)
    assert fee(m, 0.2, 100) == pytest.approx(fee(m, 0.8, 100))
    assert fee(m, 0.2, 100, maker=True) == 0.0
    # zero-fee schedule (seen on NFL game markets in Sep 2026)
    assert polymarket_fee_model({"rate": 0, "takerOnly": True}).kind is FeeKind.ZERO
    # category fallback
    assert polymarket_fee_model(None, "Crypto").taker_rate == pytest.approx(0.07)


# ------------------------------------------------------------------------------ polymarket


def test_parse_ts_variants() -> None:
    assert parse_ts("2026-09-27T17:00:00Z").hour == 17
    assert parse_ts("2026-09-09 14:25:04+00").minute == 25
    assert parse_ts(1790530421717).year == 2026
    assert parse_ts("") is None


def test_parse_gamma_moneyline_market(pm_event_kcmia: dict) -> None:
    m = next(x for x in pm_event_kcmia["markets"] if x["sportsMarketType"] == "moneyline")
    contracts = parse_gamma_market(m, pm_event_kcmia)
    assert [c.outcome for c in contracts] == ["Chiefs", "Dolphins"]
    chiefs, dolphins = contracts
    assert chiefs.venue is Venue.POLYMARKET
    assert chiefs.market_id != dolphins.market_id
    assert chiefs.parent_id == m["conditionId"]
    assert chiefs.event_title == "Chiefs vs. Dolphins"
    assert chiefs.tick_size == pytest.approx(0.01)
    assert chiefs.resolution_source == "https://www.nfl.com/scores"
    assert chiefs.close_time is not None
    # complementary tokens: bid/ask mirror
    if chiefs.best_bid is not None and dolphins.best_ask is not None:
        assert chiefs.best_bid == pytest.approx(1 - dolphins.best_ask)


def test_polymarket_status_mapping() -> None:
    base = {"active": True, "closed": False, "acceptingOrders": True}
    assert market_status(base)[0] is MarketStatus.OPEN
    assert market_status({**base, "acceptingOrders": False})[0] is MarketStatus.PAUSED
    closed = {**base, "closed": True}
    assert market_status({**closed, "umaResolutionStatus": "resolved"})[0] is MarketStatus.RESOLVED
    assert market_status({**closed, "umaResolutionStatus": "proposed"})[0] is MarketStatus.PROPOSED
    s, raw = market_status(
        {
            **base,
            "umaResolutionStatus": "disputed",
            "umaResolutionStatuses": '["proposed","disputed"]',
        }
    )
    assert s is MarketStatus.DISPUTED and "uma_history=proposed,disputed" in raw


def test_parse_resolved_polymarket_markets() -> None:
    markets = load("pm_closed_markets.json")
    resolved = [c for m in markets for c in parse_gamma_market(m)]
    assert resolved
    for c in resolved:
        assert c.status is MarketStatus.RESOLVED
        assert c.result in (0.0, 1.0)
    # the two tokens of one condition settle to complementary values
    first = parse_gamma_market(markets[0])
    assert first[0].result + first[1].result == pytest.approx(1.0)


# ------------------------------------------------------------------------------ kalshi


def test_parse_kalshi_market_with_event_and_series(kalshi_kcmia) -> None:
    markets, event, series = kalshi_kcmia
    kc = parse_market(next(m for m in markets if m["ticker"].endswith("-KC")), event, series)
    assert kc.venue is Venue.KALSHI
    assert kc.outcome == "Kansas City"
    assert kc.event_title == "Kansas City vs Miami"
    assert kc.status is MarketStatus.OPEN
    assert kc.fees.kind is FeeKind.KALSHI and kc.fees.maker_rate > 0
    assert kc.tick_size == pytest.approx(0.01)
    assert "nfl.com" in kc.resolution_source
    assert "tie" in kc.rules_text.lower()
    assert kc.best_bid is not None and kc.best_ask is not None and kc.best_bid < kc.best_ask


def test_parse_kalshi_settled_markets() -> None:
    ms = [parse_market(m) for m in load("kalshi_settled_nfl.json")["markets"]]
    assert all(m.status is MarketStatus.RESOLVED for m in ms)
    assert {m.result for m in ms} == {0.0, 1.0}


def test_parse_kalshi_orderbook_yes_no_to_bid_ask() -> None:
    bids, asks = parse_orderbook(load("kalshi_orderbook_kc.json"))
    assert bids and asks
    best_bid = max(p for p, _ in bids)
    best_ask = min(p for p, _ in asks)
    assert 0 < best_bid < best_ask < 1
    # legacy integer-cent format
    b2, a2 = parse_orderbook({"orderbook": {"yes": [[40, 10]], "no": [[55, 5]]}})
    assert b2 == [(0.4, 10.0)] and a2 == [(0.45, 5.0)]
