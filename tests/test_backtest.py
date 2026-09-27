from __future__ import annotations

import random

import pytest

from pmmon.backtest.engine import BacktestConfig, Backtester, Context, Strategy
from pmmon.backtest.strategies import ArbPair, CrossVenueArb, PassiveMarketMaker
from pmmon.fees import kalshi_fee_model
from pmmon.models import BookEventKind, BookRow, FeeModel, Side, Venue

S, D, T = BookEventKind.SNAPSHOT, BookEventKind.DELTA, BookEventKind.TRADE
PM, KA = Venue.POLYMARKET, Venue.KALSHI


def snap(v: Venue, m: str, ts: int, bids, asks) -> list[BookRow]:
    eid = f"s{ts}"
    return [BookRow(v, m, ts, ts, 0, S, Side.BID, p, q, eid) for p, q in bids] + [
        BookRow(v, m, ts, ts, 0, S, Side.ASK, p, q, eid) for p, q in asks
    ]


def delta(v: Venue, m: str, ts: int, side: Side, p: float, q: float) -> BookRow:
    return BookRow(v, m, ts, ts, 1, D, side, p, q, f"d{ts}{side}{p}")


def trade(v: Venue, m: str, ts: int, side: Side, p: float, q: float) -> BookRow:
    return BookRow(v, m, ts, ts, 1, T, side, p, q, f"t{ts}")


ZERO_LAT = BacktestConfig(latency_ms={PM: 0, KA: 0})


class OneQuote(Strategy):
    """Places a single passive bid at the start."""

    def __init__(self, price: float, size: float) -> None:
        self.price, self.size, self.done = price, size, False

    def on_book(self, ctx: Context, market: str) -> None:
        if not self.done:
            ctx.submit(market, Side.BID, self.price, self.size, "post")
            self.done = True


def test_queue_position_fill_from_trade_prints() -> None:
    rows = snap(PM, "m", 0, [(0.40, 100)], [(0.42, 100)])
    rows += [trade(PM, "m", 100, Side.BID, 0.40, 60)]  # 60 < queue ahead (100): no fill
    rows += [delta(PM, "m", 100, Side.BID, 0.40, 40)]
    rows += [trade(PM, "m", 200, Side.BID, 0.40, 50)]  # queue ahead 40 -> we get 10
    rows += [delta(PM, "m", 200, Side.BID, 0.40, 0)]
    rows += [delta(PM, "m", 300, Side.BID, 0.39, 10)]
    res = Backtester(rows, {}, OneQuote(0.40, 25), ZERO_LAT).run()
    assert res.maker_fills == 1
    assert res.fills[0].qty == pytest.approx(10)
    assert res.fills[0].price == pytest.approx(0.40)


def test_cancellations_ahead_improve_queue_position() -> None:
    rows = snap(PM, "m", 0, [(0.40, 100)], [(0.42, 100)])
    rows += [delta(PM, "m", 100, Side.BID, 0.40, 50)]  # 50 cancelled, pro-rata ahead of us
    rows += [trade(PM, "m", 200, Side.BID, 0.40, 60)]  # queue ahead 50 -> fill 10
    res = Backtester(rows, {}, OneQuote(0.40, 25), ZERO_LAT).run()
    assert sum(f.qty for f in res.fills) == pytest.approx(10)


def test_trade_through_fills_fully_and_post_only_rejects_cross() -> None:
    rows = snap(PM, "m", 0, [(0.40, 100)], [(0.42, 100)])
    rows += [delta(PM, "m", 100, Side.ASK, 0.39, 5)]  # ask moves below our bid
    res = Backtester(rows, {}, OneQuote(0.40, 25), ZERO_LAT).run()
    assert sum(f.qty for f in res.fills) == pytest.approx(25)
    res2 = Backtester(
        snap(PM, "m", 0, [(0.40, 1)], [(0.42, 1)]) + [delta(PM, "m", 5, Side.BID, 0.40, 2)],
        {},
        OneQuote(0.42, 5),
        ZERO_LAT,
    ).run()
    assert res2.n_fills == 0  # would cross -> rejected


def test_hidden_trades_for_venues_without_prints() -> None:
    rows = snap(KA, "k", 0, [(0.40, 100)], [(0.42, 100)])
    rows += [delta(KA, "k", 100, Side.BID, 0.40, 0)]  # level vanished, no trade prints
    cfg = BacktestConfig(latency_ms={KA: 0}, hidden_trade_share={KA: 0.5})
    res = Backtester(rows, {}, OneQuote(0.40, 25), cfg).run()
    # 50 of the 100 decrease are hidden trades -> they only consume the 100 ahead of us
    assert res.n_fills == 0
    cfg_all = BacktestConfig(latency_ms={KA: 0}, hidden_trade_share={KA: 1.0})
    rows2 = snap(KA, "k", 0, [(0.40, 100)], [(0.42, 100)]) + [
        delta(KA, "k", 100, Side.BID, 0.40, 0),
        delta(KA, "k", 200, Side.BID, 0.40, 30),
        delta(KA, "k", 300, Side.BID, 0.40, 0),
    ]
    res2 = Backtester(rows2, {}, OneQuote(0.40, 25), cfg_all).run()
    assert sum(f.qty for f in res2.fills) == pytest.approx(25)


def _random_walk_rows(seed: int, n: int = 3000) -> list[BookRow]:
    rng = random.Random(seed)
    rows = snap(PM, "m", 0, [(0.50, 200)], [(0.52, 200)])
    bid, ask = 0.50, 0.52
    for i in range(1, n):
        ts = i * 200
        r = rng.random()
        if r < 0.3:
            rows.append(trade(PM, "m", ts, Side.BID, bid, rng.uniform(10, 250)))
        elif r < 0.6:
            rows.append(trade(PM, "m", ts, Side.ASK, ask, rng.uniform(10, 250)))
        elif r < 0.8:
            step = rng.choice([-0.01, 0.01])
            nb, na = round(bid + step, 2), round(ask + step, 2)
            if 0.05 < nb < na < 0.95:
                rows += snap(
                    PM, "m", ts, [(nb, rng.uniform(50, 300))], [(na, rng.uniform(50, 300))]
                )
                bid, ask = nb, na
        else:
            rows.append(delta(PM, "m", ts, Side.BID, bid, rng.uniform(20, 300)))
    return rows


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_mm_attribution_identity_and_markouts(seed: int) -> None:
    fees = {"m": FeeModel(kind="polymarket", taker_rate=0.05, maker_rate=0.01)}
    strat = PassiveMarketMaker(
        markets=["m"], size=20, half_spread=0.005, max_inventory=200, skew=0.02, min_requote_ms=0
    )
    res = Backtester(_random_walk_rows(seed), fees, strat, ZERO_LAT).run()
    assert res.maker_fills > 0 and res.quotes_placed > 0
    parts = res.spread_capture + res.inventory_pnl - res.fees
    assert res.total_pnl == pytest.approx(parts, abs=1e-9)
    assert res.fees > 0
    assert set(res.markouts) == {1_000, 10_000, 60_000}
    assert res.max_abs_inventory <= 200 + 20


def _arb_rows(move_at: int) -> list[BookRow]:
    rows = snap(PM, "a", 0, [(0.38, 100)], [(0.40, 100)])
    rows += snap(KA, "b", 10, [(0.45, 100)], [(0.47, 100)])
    # Polymarket ask gets lifted by someone else shortly after the signal
    rows += [
        delta(PM, "a", move_at, Side.ASK, 0.40, 0),
        delta(PM, "a", move_at, Side.ASK, 0.46, 50),
    ]
    rows += [delta(KA, "b", 5_000, Side.ASK, 0.47, 90)]
    return rows


def test_arb_with_zero_latency_locks_edge() -> None:
    strat = CrossVenueArb(pairs=[ArbPair("a", "b")], size=100, min_edge=0.01)
    res = Backtester(_arb_rows(100), {}, strat, ZERO_LAT).run()
    st = strat.leg_stats(res.fills)
    assert st["both_legs_filled"] == 1
    assert st["locked_pnl_if_same_resolution_usd"] == pytest.approx(100 * (0.45 - 0.40))


def test_arb_latency_leaves_one_leg() -> None:
    strat = CrossVenueArb(pairs=[ArbPair("a", "b")], size=100, min_edge=0.01)
    cfg = BacktestConfig(latency_ms={PM: 250, KA: 150})
    res = Backtester(_arb_rows(100), {}, strat, cfg).run()
    st = strat.leg_stats(res.fills)
    assert st["both_legs_filled"] == 0 and st["partial_or_one_leg"] == 1
    assert st["unhedged_contracts"] == pytest.approx(100)


def test_arb_fees_remove_thin_edge() -> None:
    rows = snap(PM, "a", 0, [(0.48, 100)], [(0.50, 100)]) + snap(
        KA, "b", 10, [(0.51, 100)], [(0.53, 100)]
    )
    fees = {"b": kalshi_fee_model("quadratic")}  # ~1.75c per contract at p=0.5
    strat = CrossVenueArb(pairs=[ArbPair("a", "b")], size=100, min_edge=0.0)
    Backtester(rows, fees, strat, ZERO_LAT).run()
    assert strat.signals == 0  # 1c gross edge < 1.75c Kalshi taker fee
    strat2 = CrossVenueArb(pairs=[ArbPair("a", "b")], size=100, min_edge=0.0)
    Backtester(rows, {}, strat2, ZERO_LAT).run()
    assert strat2.signals >= 1


def test_polled_venue_ioc_needs_liquidity_to_survive_confirmation_window() -> None:
    rows = snap(PM, "a", 0, [(0.38, 100)], [(0.40, 100)])
    rows += snap(KA, "b", 10, [(0.45, 100)], [(0.47, 100)])
    rows += [delta(KA, "b", 1_000, Side.BID, 0.45, 0), delta(KA, "b", 1_000, Side.BID, 0.41, 100)]
    rows += [delta(PM, "a", 10_000, Side.BID, 0.37, 5)]
    strat = CrossVenueArb(pairs=[ArbPair("a", "b")], size=100, min_edge=0.01)
    res = Backtester(rows, {}, strat, ZERO_LAT).run()  # default: Kalshi confirm 3 s
    st = strat.leg_stats(res.fills)
    assert st["partial_or_one_leg"] == 1 and st["unhedged_contracts"] == pytest.approx(100)
    naive = BacktestConfig(latency_ms={PM: 0, KA: 0}, ioc_confirm_ms={KA: 0})
    strat2 = CrossVenueArb(pairs=[ArbPair("a", "b")], size=100, min_edge=0.01)
    res2 = Backtester(rows, {}, strat2, naive).run()
    assert strat2.leg_stats(res2.fills)["both_legs_filled"] == 1
