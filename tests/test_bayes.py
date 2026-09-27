from __future__ import annotations

import math
import random

import pytest

from pmmon.bayes import (
    BetaBinomial,
    calibration_report,
    edge_vs_market,
    logit,
    logodds_update,
    sigmoid,
)


def test_beta_binomial_update_and_moments() -> None:
    prior = BetaBinomial.from_mean(0.3, strength=10)
    assert prior.alpha == pytest.approx(3) and prior.beta == pytest.approx(7)
    post = prior.update(successes=7, failures=3)
    assert post.mean == pytest.approx(10 / 20)
    assert post.var < prior.var


def test_beta_cdf_against_numeric_integration() -> None:
    b = BetaBinomial(2.5, 4.0)
    n = 20_000
    norm = math.exp(math.lgamma(6.5) - math.lgamma(2.5) - math.lgamma(4.0))
    x = 0.35
    h = x / n
    integral = sum(norm * ((i + 0.5) * h) ** 1.5 * (1 - (i + 0.5) * h) ** 3 for i in range(n)) * h
    assert b.cdf(x) == pytest.approx(integral, abs=1e-6)
    lo, hi = b.interval(0.9)
    assert b.cdf(lo) == pytest.approx(0.05, abs=1e-6) and b.cdf(hi) == pytest.approx(0.95, abs=1e-6)


def test_beta_binomial_predictive_sums_to_one() -> None:
    b = BetaBinomial(2, 3)
    assert sum(b.predictive_prob(k, 10) for k in range(11)) == pytest.approx(1.0)


def test_logodds_update() -> None:
    assert sigmoid(logit(0.37)) == pytest.approx(0.37)
    path = logodds_update(0.5, [3.0, 3.0])
    assert path[0] == pytest.approx(0.75) and path[1] == pytest.approx(0.9)
    tempered = logodds_update(0.5, [3.0, 3.0], temper=0.5)
    assert tempered[-1] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        logodds_update(0.5, [0.0])


def test_edge_vs_market() -> None:
    assert edge_vs_market(0.6, 0.50, 0.52) == ("buy_yes", pytest.approx(0.08))
    assert edge_vs_market(0.4, 0.50, 0.52) == ("sell_yes", pytest.approx(0.10))
    assert edge_vs_market(0.51, 0.50, 0.52)[0] == "none"


def test_calibration_perfect_vs_overconfident() -> None:
    rng = random.Random(7)
    preds = [rng.random() for _ in range(20_000)]
    outs = [1.0 if rng.random() < p else 0.0 for p in preds]
    good = calibration_report(preds, outs)
    assert good.ece < 0.02
    # Murphy decomposition: brier = reliability - resolution + uncertainty (binned approx.)
    assert good.brier == pytest.approx(
        good.reliability - good.resolution + good.uncertainty, abs=5e-3
    )
    over = calibration_report([min(max(0.5 + 1.8 * (p - 0.5), 0), 1) for p in preds], outs)
    assert over.ece > good.ece and over.brier > good.brier
    assert "0.9-1.0" in good.table()


def test_calibration_validates_input() -> None:
    with pytest.raises(ValueError):
        calibration_report([], [])
