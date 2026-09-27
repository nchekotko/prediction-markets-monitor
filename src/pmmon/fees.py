"""Trading-fee arithmetic for both venues.

Polymarket (docs.polymarket.com/trading/fees, per-market ``feeSchedule`` in Gamma):
    fee = C * rate * (p * (1 - p)) ** exponent, rounded to 5 decimals (min 0.00001 USDC);
    makers pay nothing (``takerOnly``).

Kalshi (fee schedule; series ``fee_type`` / ``fee_multiplier``):
    taker = ceil_to_cent(0.07 * multiplier * C * p * (1 - p))
    maker = ceil_to_cent(0.0175 * multiplier * C * p * (1 - p))  only for
    ``quadratic_with_maker_fees`` series, otherwise 0.
"""

from __future__ import annotations

import math

from pmmon.models import FeeKind, FeeModel

KALSHI_TAKER_RATE = 0.07
KALSHI_MAKER_RATE = 0.0175

# Fallback taker rates when a Polymarket market carries no explicit feeSchedule.
POLYMARKET_CATEGORY_RATES: dict[str, float] = {
    "crypto": 0.07,
    "sports": 0.05,
    "finance": 0.04,
    "politics": 0.04,
    "mentions": 0.04,
    "tech": 0.04,
    "economics": 0.05,
    "culture": 0.05,
    "weather": 0.05,
    "other": 0.05,
    "geopolitics": 0.0,
}


def _ceil_cent(x: float) -> float:
    # subtract a tiny epsilon so exact cents do not round up because of float noise
    return math.ceil(x * 100 - 1e-9) / 100


def fee(model: FeeModel, price: float, contracts: float, *, maker: bool = False) -> float:
    """Fee in USD for trading ``contracts`` at ``price`` (probability units)."""
    if contracts <= 0 or model.kind is FeeKind.ZERO:
        return 0.0
    p = min(max(price, 0.0), 1.0)
    rate = model.maker_rate if maker else model.taker_rate
    if rate <= 0:
        return 0.0
    if model.kind is FeeKind.POLYMARKET:
        raw = contracts * rate * (p * (1 - p)) ** model.exponent
        return round(raw, 5)
    if model.kind is FeeKind.KALSHI:
        raw = rate * model.multiplier * contracts * p * (1 - p)
        return _ceil_cent(raw)
    raise ValueError(f"unknown fee kind {model.kind}")


def fee_per_contract(
    model: FeeModel, price: float, contracts: float, *, maker: bool = False
) -> float:
    return fee(model, price, contracts, maker=maker) / contracts if contracts > 0 else 0.0


def polymarket_fee_model(
    fee_schedule: dict | None, category: str | None = None, fees_enabled: bool | None = None
) -> FeeModel:
    if fees_enabled is False:
        return FeeModel()
    if fee_schedule:
        rate = float(fee_schedule.get("rate") or 0.0)
        if rate <= 0:
            return FeeModel()
        maker_rate = 0.0 if fee_schedule.get("takerOnly", True) else rate
        return FeeModel(
            kind=FeeKind.POLYMARKET,
            taker_rate=rate,
            maker_rate=maker_rate,
            exponent=float(fee_schedule.get("exponent") or 1.0),
        )
    if category is not None:
        rate = POLYMARKET_CATEGORY_RATES.get(category.lower(), 0.0)
        if rate > 0:
            return FeeModel(kind=FeeKind.POLYMARKET, taker_rate=rate)
    return FeeModel()


def kalshi_fee_model(fee_type: str | None, fee_multiplier: float | None = None) -> FeeModel:
    mult = float(fee_multiplier) if fee_multiplier is not None else 1.0
    ft = (fee_type or "quadratic").lower()
    # Unknown fee types are treated as the standard quadratic schedule (conservative).
    if ft in {"zero", "none", "free"} or mult <= 0:
        return FeeModel()
    maker = KALSHI_MAKER_RATE if ft == "quadratic_with_maker_fees" else 0.0
    return FeeModel(
        kind=FeeKind.KALSHI, taker_rate=KALSHI_TAKER_RATE, maker_rate=maker, multiplier=mult
    )
