"""Unified, venue-agnostic data model.

Every tradable *binary contract* (one outcome token on Polymarket, one market ticker on
Kalshi) becomes one :class:`UnifiedMarket`. Prices are always probability units in [0, 1]
for the YES side of that contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Venue(StrEnum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


class ContractType(StrEnum):
    BINARY = "binary"


class MarketStatus(StrEnum):
    UNOPENED = "unopened"
    OPEN = "open"
    PAUSED = "paused"
    CLOSED = "closed"  # trading stopped, outcome not yet final
    PROPOSED = "proposed"  # outcome proposed (UMA) / determined (Kalshi), dispute window open
    DISPUTED = "disputed"
    RESOLVED = "resolved"
    VOID = "void"


class FeeKind(StrEnum):
    ZERO = "zero"
    POLYMARKET = "polymarket"  # fee = C * rate * (p * (1 - p)) ** exponent, rounded to 1e-5 USDC
    KALSHI = "kalshi"  # fee = ceil_to_cent(rate * multiplier * C * p * (1 - p))


class FeeModel(BaseModel):
    """Per-contract fee schedule; see :mod:`pmmon.fees` for the arithmetic."""

    model_config = ConfigDict(frozen=True)

    kind: FeeKind = FeeKind.ZERO
    taker_rate: float = 0.0
    maker_rate: float = 0.0
    exponent: float = 1.0
    multiplier: float = 1.0


class UnifiedMarket(BaseModel):
    """One binary contract, normalised across venues."""

    venue: Venue
    market_id: str = Field(description="Book key: Polymarket CLOB token id / Kalshi market ticker")
    parent_id: str = Field(description="Polymarket condition id / Kalshi market ticker")
    event_id: str
    event_title: str
    question: str
    outcome: str = Field(description="Human label of what YES means for this contract")
    contract_type: ContractType = ContractType.BINARY
    status: MarketStatus
    best_bid: float | None = None
    best_ask: float | None = None
    last_price: float | None = None
    tick_size: float = 0.01
    min_order_size: float | None = None
    fees: FeeModel = FeeModel()
    open_time: datetime | None = None
    close_time: datetime | None = None
    expected_resolution_time: datetime | None = None
    resolution_source: str = ""
    rules_text: str = ""
    result: float | None = Field(default=None, description="Settlement value of YES in [0,1]")
    volume: float | None = None
    category: str | None = None
    url: str | None = None
    raw_status: str = ""

    @field_validator("best_bid", "best_ask", "last_price", "result")
    @classmethod
    def _prob_units(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"price {v} is not in probability units [0, 1]")
        return v

    @property
    def key(self) -> str:
        return f"{self.venue.value}:{self.market_id}"


class Side(StrEnum):
    BID = "bid"
    ASK = "ask"


class BookEventKind(StrEnum):
    SNAPSHOT = "snapshot"
    DELTA = "delta"
    TRADE = "trade"


@dataclass(frozen=True, slots=True)
class BookRow:
    """One persisted row of the normalised L2 stream.

    * snapshot: one row per level; all rows of a snapshot share ``event_id``;
    * delta: absolute new size at ``price`` on ``side`` (0 removes the level);
    * trade: ``side`` is the side of the *resting* order that was hit (bid = seller aggressed).

    The idempotency key is ``(venue, market_id, event_id, kind, side, price)``.
    A plain slotted dataclass (not Pydantic) because millions of these flow through the
    recorder and the backtester.
    """

    venue: Venue
    market_id: str
    ts_ms: int  # exchange timestamp (or poll time when the venue gives none)
    recv_ts_ms: int
    seq: int  # venue sequence number when available, else per-market local counter
    kind: BookEventKind
    side: Side
    price: float
    size: float
    event_id: str

    @property
    def dedup_key(self) -> tuple[str, str, str, str, str, float]:
        return (
            self.venue.value,
            self.market_id,
            self.event_id,
            self.kind.value,
            self.side.value,
            self.price,
        )
