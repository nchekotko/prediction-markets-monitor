"""Bayesian probability updating and calibration of market prices.

* :class:`BetaBinomial` - conjugate Beta prior over a rate, updated with success/failure
  counts; the market price can be used as the prior mean with an "equivalent sample size".
* :func:`logodds_update` - sequential updating of a probability with signals expressed as
  likelihood ratios (or log-likelihood ratios), optionally tempered for correlated signals.
* :func:`calibration_report` - Brier score, log loss, Murphy decomposition
  (reliability / resolution / uncertainty), expected calibration error and a reliability
  table for (price, outcome) pairs from resolved markets.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

EPS = 1e-6


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


@dataclass(frozen=True)
class BetaBinomial:
    alpha: float
    beta: float

    @classmethod
    def from_mean(cls, mean: float, strength: float) -> BetaBinomial:
        """Prior with given mean (e.g. market price) worth ``strength`` pseudo-observations."""
        if not 0 < mean < 1 or strength <= 0:
            raise ValueError("mean must be in (0,1) and strength > 0")
        return cls(mean * strength, (1 - mean) * strength)

    def update(self, successes: float, failures: float) -> BetaBinomial:
        return BetaBinomial(self.alpha + successes, self.beta + failures)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def var(self) -> float:
        a, b = self.alpha, self.beta
        return a * b / ((a + b) ** 2 * (a + b + 1))

    def interval(self, level: float = 0.9) -> tuple[float, float]:
        """Equal-tailed credible interval via bisection on the regularized incomplete beta."""
        lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
        return self.ppf(lo_q), self.ppf(hi_q)

    def cdf(self, x: float) -> float:
        return _betainc(self.alpha, self.beta, x)

    def ppf(self, q: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if self.cdf(mid) < q:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def predictive_prob(self, k: int, n: int) -> float:
        """Beta-binomial P(K = k | n)."""
        a, b = self.alpha, self.beta
        return math.exp(
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + _lbeta(k + a, n - k + b)
            - _lbeta(a, b)
        )


def _lbeta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) via Lentz continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1 - _betainc(b, a, 1 - x)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - _lbeta(a, b)) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(400):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1 + num * d
        d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1 + num / c if abs(c) > 1e-30 else 1 + num / 1e-30
        f *= c * d
        if abs(c * d - 1) < 1e-12:
            break
    return front * (f - 1)


def logodds_update(
    prior: float, likelihood_ratios: Iterable[float], temper: float = 1.0
) -> list[float]:
    """Posterior path after each signal: logit(p) += temper * log(LR).

    ``temper`` < 1 discounts signals that are not conditionally independent.
    """
    x = logit(prior)
    path = []
    for lr in likelihood_ratios:
        if lr <= 0:
            raise ValueError("likelihood ratio must be positive")
        x += temper * math.log(lr)
        path.append(sigmoid(x))
    return path


def edge_vs_market(posterior: float, bid: float, ask: float) -> tuple[str, float]:
    """Expected value per contract of the better side vs executable prices (before fees)."""
    buy = posterior - ask
    sell = bid - posterior
    if buy >= sell and buy > 0:
        return "buy_yes", buy
    if sell > 0:
        return "sell_yes", sell
    return "none", max(buy, sell)


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    n: int
    mean_pred: float
    freq: float


@dataclass
class CalibrationReport:
    n: int
    brier: float
    log_loss: float
    reliability: float
    resolution: float
    uncertainty: float
    ece: float
    base_rate: float
    bins: list[ReliabilityBin] = field(default_factory=list)

    def table(self) -> str:
        lines = ["bin         n   mean_pred  freq    |"]
        for b in self.bins:
            if b.n == 0:
                continue
            bar = "#" * round(b.freq * 20)
            mark = round(b.mean_pred * 20)
            row = list(bar.ljust(21))
            row[min(mark, 20)] = "|"
            lines.append(
                f"{b.lo:.1f}-{b.hi:.1f}  {b.n:5d}   {b.mean_pred:.3f}    {b.freq:.3f}"
                f"   {''.join(row)}"
            )
        return "\n".join(lines)

    def summary(self) -> dict[str, float]:
        return {
            "n": self.n,
            "brier": round(self.brier, 5),
            "log_loss": round(self.log_loss, 5),
            "reliability": round(self.reliability, 5),
            "resolution": round(self.resolution, 5),
            "uncertainty": round(self.uncertainty, 5),
            "ece": round(self.ece, 5),
            "base_rate": round(self.base_rate, 4),
        }


def calibration_report(
    preds: Sequence[float], outcomes: Sequence[float], n_bins: int = 10
) -> CalibrationReport:
    if len(preds) != len(outcomes) or not preds:
        raise ValueError("need equally sized, non-empty inputs")
    n = len(preds)
    brier = sum((p - o) ** 2 for p, o in zip(preds, outcomes, strict=True)) / n
    ll = (
        -sum(
            o * math.log(min(max(p, EPS), 1 - EPS))
            + (1 - o) * math.log(1 - min(max(p, EPS), 1 - EPS))
            for p, o in zip(preds, outcomes, strict=True)
        )
        / n
    )
    base = sum(outcomes) / n
    bins: list[ReliabilityBin] = []
    rel = res = ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [j for j, p in enumerate(preds) if lo <= p < hi or (i == n_bins - 1 and p == 1.0)]
        if not idx:
            bins.append(ReliabilityBin(lo, hi, 0, float("nan"), float("nan")))
            continue
        mp = sum(preds[j] for j in idx) / len(idx)
        fr = sum(outcomes[j] for j in idx) / len(idx)
        w = len(idx) / n
        rel += w * (mp - fr) ** 2
        res += w * (fr - base) ** 2
        ece += w * abs(mp - fr)
        bins.append(ReliabilityBin(lo, hi, len(idx), mp, fr))
    return CalibrationReport(n, brier, ll, rel, res, base * (1 - base), ece, base, bins)
