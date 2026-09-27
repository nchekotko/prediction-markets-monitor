from __future__ import annotations

from datetime import timedelta

import pytest
from conftest import load

from pmmon.book import L2Book
from pmmon.fees import kalshi_fee_model, polymarket_fee_model
from pmmon.models import BookEventKind, BookRow, FeeModel, MarketStatus, Side, Venue
from pmmon.monitor.alerts import BookAlertRules, ResolutionMonitor, unusual_rule_terms
from pmmon.monitor.metrics import arb_for_size, best_arb, book_metrics
from pmmon.monitor.replay import PairSpec, replay_monitor
from pmmon.venues.kalshi import parse_market


def _book(m: str, bids, asks, recv: int = 0) -> L2Book:
    b = L2Book(m)
    b.reset(bids, asks)
    b.last_update_recv_ms = recv
    return b


def test_book_metrics() -> None:
    b = _book("m", [(0.60, 300), (0.59, 100)], [(0.62, 100)], recv=1000)
    mt = book_metrics(b, now_ms=4000, depth_window=0.01)
    assert mt.spread == pytest.approx(0.02) and mt.mid == pytest.approx(0.61)
    assert mt.microprice == pytest.approx((0.60 * 100 + 0.62 * 300) / 400)
    assert mt.bid_depth == 400 and mt.imbalance == pytest.approx(0.5)
    assert mt.age_ms == 3000


def test_arb_net_of_fees_and_slippage_walks_depth() -> None:
    a = _book("pm", [(0.38, 500)], [(0.40, 60), (0.41, 100)])
    b = _book("k", [(0.45, 80), (0.44, 100)], [(0.47, 100)])
    zero = FeeModel()
    q = arb_for_size(a, zero, b, zero, 100)
    assert q.size == 100
    assert q.buy_vwap == pytest.approx((60 * 0.40 + 40 * 0.41) / 100)
    assert q.sell_vwap == pytest.approx((80 * 0.45 + 20 * 0.44) / 100)
    assert q.net_pnl == pytest.approx(100 * (q.sell_vwap - q.buy_vwap))
    kf = kalshi_fee_model("quadratic")
    pf = polymarket_fee_model({"rate": 0.05, "exponent": 1, "takerOnly": True})
    q2 = arb_for_size(a, pf, b, kf, 100, slippage_per_contract=0.001)
    assert q2.fees > 0 and q2.net_pnl < q.net_pnl - q2.fees + 1e-9
    assert best_arb(b, zero, a, zero, 100).buy_market == "pm"  # direction auto-detected


def test_book_alert_rules() -> None:
    rules = BookAlertRules(stale_after_ms=10_000, wide_spread=0.1)
    b = _book("m", [(0.2, 1)], [(0.35, 1)], recv=1)
    kinds = {a.kind for a in rules.check_book(b, 20_000, "m")}
    assert kinds == {"stale_book", "wide_spread"}
    crossed = _book("x", [(0.5, 1)], [(0.5, 1)], recv=20_000)
    assert "crossed_book" in {a.kind for a in rules.check_book(crossed, 20_000, "x")}


def _kalshi_kc():
    ev = load("kalshi_event_kcmia.json")["event"]
    m = next(x for x in load("kalshi_markets_kcmia.json")["markets"] if x["ticker"].endswith("KC"))
    return parse_market(m, ev)


def test_resolution_monitor_lifecycle() -> None:
    mon = ResolutionMonitor(warn_before=timedelta(hours=6))
    kc = _kalshi_kc()
    close = kc.close_time
    assert close is not None
    a0 = mon.update([kc], close - timedelta(days=1))
    assert a0 == []
    a1 = mon.update([kc], close - timedelta(hours=2))
    assert [a.kind for a in a1] == ["closing_soon"]
    assert mon.update([kc], close - timedelta(hours=1)) == []  # warned once
    closed = kc.model_copy(update={"status": MarketStatus.CLOSED})
    a2 = mon.update([closed], close + timedelta(minutes=5))
    assert a2[0].kind == "status_change" and a2[0].data["to"] == "closed"
    disputed = kc.model_copy(update={"status": MarketStatus.DISPUTED})
    a3 = mon.update([disputed], close + timedelta(hours=1))
    assert a3[0].severity.value == "critical"
    tie = kc.model_copy(
        update={
            "status": MarketStatus.RESOLVED,
            "result": 0.5,
            "raw_status": "uma=resolved;uma_history=proposed,disputed",
        }
    )
    kinds = {a.kind for a in mon.update([tie], close + timedelta(hours=3))}
    assert {"status_change", "non_binary_resolution", "resolved_after_dispute"} <= kinds


def test_open_past_close_alert() -> None:
    kc = _kalshi_kc()
    mon = ResolutionMonitor()
    alerts = mon.update([kc], kc.close_time + timedelta(hours=2))
    assert any(a.kind == "open_past_close" for a in alerts)


def test_unusual_rule_terms_on_real_rules() -> None:
    kc = _kalshi_kc()
    terms = unusual_rule_terms(kc.rules_text)
    assert "tie" in terms and "fair price" in terms and "postpone" not in terms


def test_replay_monitor_finds_episode() -> None:
    rows: list[BookRow] = []

    def snap(v, m, ts, bids, asks):
        return [
            BookRow(v, m, ts, ts, 0, BookEventKind.SNAPSHOT, Side.BID, p, q, f"s{ts}")
            for p, q in bids
        ] + [
            BookRow(v, m, ts, ts, 0, BookEventKind.SNAPSHOT, Side.ASK, p, q, f"s{ts}")
            for p, q in asks
        ]

    rows += snap(Venue.POLYMARKET, "a", 0, [(0.38, 100)], [(0.40, 100)])
    rows += snap(Venue.KALSHI, "b", 500, [(0.39, 100)], [(0.41, 100)])
    rows += snap(Venue.KALSHI, "b", 2_000, [(0.43, 100)], [(0.45, 100)])  # crosses PM ask
    rows += snap(Venue.KALSHI, "b", 5_000, [(0.39, 100)], [(0.41, 100)])  # gone
    rows += snap(Venue.POLYMARKET, "a", 100_000, [(0.38, 100)], [(0.40, 100)])
    spec = PairSpec("a", "b", FeeModel(), FeeModel(), "a~b")
    stats, alerts = replay_monitor(
        rows, [spec], size=100, rules=BookAlertRules(stale_after_ms=30_000)
    )
    s = stats[0].summary()
    assert s["arb_episodes"] == 1 and s["median_episode_s"] == pytest.approx(3.0)
    assert s["max_net_pnl_usd"] == pytest.approx(3.0)
    kinds = {a.kind for a in alerts}
    assert "cross_venue_arb" in kinds and "stale_book" in kinds
