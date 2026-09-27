"""Book metrics and cross-venue divergence net of fees and slippage."""

from __future__ import annotations

from dataclasses import dataclass

from pmmon.book import L2Book
from pmmon.fees import fee
from pmmon.models import FeeModel, Side


@dataclass(frozen=True)
class BookMetrics:
    market_id: str
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    mid: float | None
    microprice: float | None
    bid_depth: float
    ask_depth: float
    imbalance: float | None  # (bid - ask) / (bid + ask) at the touch
    age_ms: int


def book_metrics(book: L2Book, now_ms: int, depth_window: float = 0.05) -> BookMetrics:
    bb, bs, ba, as_ = book.top()
    imb = (bs - as_) / (bs + as_) if bs + as_ > 0 else None
    return BookMetrics(
        market_id=book.market_id,
        best_bid=bb,
        best_ask=ba,
        spread=book.spread,
        mid=book.mid,
        microprice=book.microprice,
        bid_depth=book.depth(Side.BID, depth_window),
        ask_depth=book.depth(Side.ASK, depth_window),
        imbalance=imb,
        age_ms=now_ms - book.last_update_recv_ms if book.last_update_recv_ms else -1,
    )


@dataclass(frozen=True)
class ArbQuote:
    """Buy YES on ``buy_venue`` and buy NO (= sell YES) on ``sell_venue`` for ``size``.

    Pays 1.0 per contract pair at resolution regardless of outcome *if both contracts
    resolve identically* (see resolution-risk flags from the mapper).
    """

    buy_market: str
    sell_market: str
    size: float  # contracts actually executable on both legs
    buy_vwap: float  # YES price paid
    sell_vwap: float  # YES price received (NO bought at 1 - sell_vwap)
    gross_edge: float  # per contract, before fees
    fees: float  # total USD fees for both legs
    slippage_buffer: float  # USD
    net_pnl: float  # USD for ``size`` contracts
    net_edge: float  # per contract

    @property
    def is_opportunity(self) -> bool:
        return self.size > 0 and self.net_pnl > 0


def arb_for_size(
    buy_book: L2Book,
    buy_fees: FeeModel,
    sell_book: L2Book,
    sell_fees: FeeModel,
    size: float,
    slippage_per_contract: float = 0.0,
) -> ArbQuote:
    """Evaluate the "buy YES here, sell YES there" trade by sweeping both books."""
    filled_buy, vwap_buy = buy_book.sweep(Side.ASK, size)
    filled_sell, vwap_sell = sell_book.sweep(Side.BID, size)
    q = min(filled_buy, filled_sell)
    if q <= 0:
        return ArbQuote(
            buy_book.market_id,
            sell_book.market_id,
            0.0,
            float("nan"),
            float("nan"),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    if q < size:  # re-sweep both legs at the common executable size
        _, vwap_buy = buy_book.sweep(Side.ASK, q)
        _, vwap_sell = sell_book.sweep(Side.BID, q)
    gross = vwap_sell - vwap_buy
    # Selling YES on the second venue is implemented as buying NO at (1 - price); both venues'
    # quadratic fee is symmetric in p <-> 1 - p, so fee(p) == fee(1 - p).
    fees = fee(buy_fees, vwap_buy, q) + fee(sell_fees, vwap_sell, q)
    slip = slippage_per_contract * q * 2
    net = gross * q - fees - slip
    return ArbQuote(
        buy_book.market_id,
        sell_book.market_id,
        q,
        vwap_buy,
        vwap_sell,
        gross,
        fees,
        slip,
        net,
        net / q,
    )


def best_arb(
    book_a: L2Book,
    fees_a: FeeModel,
    book_b: L2Book,
    fees_b: FeeModel,
    size: float,
    slippage_per_contract: float = 0.0,
) -> ArbQuote:
    """Check both directions and return the better one."""
    ab = arb_for_size(book_a, fees_a, book_b, fees_b, size, slippage_per_contract)
    ba = arb_for_size(book_b, fees_b, book_a, fees_a, size, slippage_per_contract)
    return ab if ab.net_pnl >= ba.net_pnl else ba


def mid_divergence(book_a: L2Book, book_b: L2Book) -> float | None:
    ma, mb = book_a.mid, book_b.mid
    return None if ma is None or mb is None else ma - mb
