"""In-memory L2 order book for one binary contract (YES side, probability units)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from pmmon.models import Side

PRICE_DECIMALS = 6


def _k(price: float) -> float:
    return round(price, PRICE_DECIMALS)


@dataclass
class L2Book:
    market_id: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    ts_ms: int = 0
    last_update_recv_ms: int = 0
    seq: int = 0

    # ---- mutation -------------------------------------------------------------------
    def reset(
        self,
        bids: Iterable[tuple[float, float]],
        asks: Iterable[tuple[float, float]],
        ts_ms: int = 0,
    ) -> None:
        self.bids = {_k(p): s for p, s in bids if s > 0}
        self.asks = {_k(p): s for p, s in asks if s > 0}
        self.ts_ms = ts_ms

    def set_level(self, side: Side, price: float, size: float) -> None:
        levels = self.bids if side is Side.BID else self.asks
        p = _k(price)
        if size <= 0:
            levels.pop(p, None)
        else:
            levels[p] = size

    # ---- read -----------------------------------------------------------------------
    @property
    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    def top(self) -> tuple[float | None, float, float | None, float]:
        bb, ba = self.best_bid, self.best_ask
        return (
            bb,
            self.bids.get(bb, 0.0) if bb is not None else 0.0,
            ba,
            (self.asks.get(ba, 0.0) if ba is not None else 0.0),
        )

    @property
    def is_crossed(self) -> bool:
        bb, ba = self.best_bid, self.best_ask
        return bb is not None and ba is not None and bb >= ba

    @property
    def spread(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        return None if bb is None or ba is None else round(ba - bb, PRICE_DECIMALS)

    @property
    def mid(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        return None if bb is None or ba is None else (bb + ba) / 2

    @property
    def microprice(self) -> float | None:
        """Size-weighted mid: leans toward the side with *less* resting size."""
        bb, bs, ba, as_ = self.top()
        if bb is None or ba is None or bs + as_ <= 0:
            return None
        return (bb * as_ + ba * bs) / (bs + as_)

    def depth(self, side: Side, within: float) -> float:
        """Contracts resting within ``within`` (probability units) of the touch."""
        if side is Side.BID:
            bb = self.best_bid
            if bb is None:
                return 0.0
            return sum(s for p, s in self.bids.items() if p >= bb - within - 1e-12)
        ba = self.best_ask
        if ba is None:
            return 0.0
        return sum(s for p, s in self.asks.items() if p <= ba + within + 1e-12)

    def sorted_levels(self, side: Side) -> list[tuple[float, float]]:
        if side is Side.BID:
            return sorted(self.bids.items(), key=lambda x: -x[0])
        return sorted(self.asks.items())

    def sweep(self, side: Side, size: float) -> tuple[float, float]:
        """Walk the book to take ``size`` contracts from ``side`` (ASK = buy, BID = sell).

        Returns ``(filled, vwap)``; ``filled < size`` when the book is too thin.
        """
        filled = 0.0
        notional = 0.0
        for price, avail in self.sorted_levels(side):
            take = min(avail, size - filled)
            filled += take
            notional += take * price
            if filled >= size - 1e-12:
                break
        return filled, (notional / filled if filled > 0 else float("nan"))

    def copy(self) -> L2Book:
        return L2Book(
            self.market_id,
            dict(self.bids),
            dict(self.asks),
            self.ts_ms,
            self.last_update_recv_ms,
            self.seq,
        )
