"""Alert rules: stale / wide / crossed books, cross-venue arbitrage, resolution lifecycle."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from pmmon.book import L2Book
from pmmon.models import MarketStatus, UnifiedMarket
from pmmon.monitor.metrics import ArbQuote


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Alert:
    ts_ms: int
    kind: str
    severity: Severity
    key: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts_ms": self.ts_ms,
            "kind": self.kind,
            "severity": self.severity.value,
            "key": self.key,
            "message": self.message,
            **self.data,
        }


@dataclass
class BookAlertRules:
    stale_after_ms: int = 30_000
    wide_spread: float = 0.10
    min_arb_net_usd: float = 0.5

    def check_book(self, book: L2Book, now_ms: int, key: str) -> list[Alert]:
        out: list[Alert] = []
        age = now_ms - book.last_update_recv_ms if book.last_update_recv_ms else None
        if age is not None and age > self.stale_after_ms:
            out.append(
                Alert(
                    now_ms,
                    "stale_book",
                    Severity.WARN,
                    key,
                    f"no book update for {age / 1000:.0f}s",
                    {"age_ms": age},
                )
            )
        if book.is_crossed:
            out.append(
                Alert(
                    now_ms,
                    "crossed_book",
                    Severity.WARN,
                    key,
                    f"crossed book bid={book.best_bid} ask={book.best_ask}",
                )
            )
        sp = book.spread
        if sp is not None and sp >= self.wide_spread:
            out.append(
                Alert(now_ms, "wide_spread", Severity.INFO, key, f"spread {sp:.3f}", {"spread": sp})
            )
        if book.best_bid is None or book.best_ask is None:
            out.append(
                Alert(now_ms, "one_sided_book", Severity.INFO, key, "book has an empty side")
            )
        return out

    def check_arb(
        self, q: ArbQuote, now_ms: int, pair_key: str, risk_flags: list[str]
    ) -> list[Alert]:
        if not q.is_opportunity or q.net_pnl < self.min_arb_net_usd:
            return []
        sev = Severity.WARN if risk_flags else Severity.CRITICAL
        msg = (
            f"buy YES {q.buy_market[:16]}@{q.buy_vwap:.3f} / sell YES {q.sell_market[:16]}"
            f"@{q.sell_vwap:.3f} x{q.size:.0f}: net ${q.net_pnl:.2f} after fees ${q.fees:.2f}"
        )
        if risk_flags:
            msg += f" [resolution risk: {', '.join(risk_flags)}]"
        return [
            Alert(
                now_ms,
                "cross_venue_arb",
                sev,
                pair_key,
                msg,
                {"net_pnl": q.net_pnl, "size": q.size, "net_edge": q.net_edge},
            )
        ]


_UNUSUAL_WORDING = re.compile(
    r"\b(50[-/ ]50|fair price|void|cancel(l)?ed|postpone|annul|tie|dispute|clarification)\b",
    re.IGNORECASE,
)


@dataclass
class ResolutionMonitor:
    """Tracks market lifecycle between successive metadata polls and raises alerts for:

    * markets approaching close (``warn_before``),
    * status transitions (open -> closed -> proposed -> resolved, anything -> disputed),
    * unusual resolutions (non-binary settlement such as 0.5, void, disputes in history),
    * trading after scheduled close / resolved without passing through close.
    """

    warn_before: timedelta = timedelta(hours=6)
    state: dict[str, UnifiedMarket] = field(default_factory=dict)
    close_warned: set[str] = field(default_factory=set)

    def update(self, markets: list[UnifiedMarket], now: datetime) -> list[Alert]:
        ts = int(now.timestamp() * 1000)
        out: list[Alert] = []
        for m in markets:
            prev = self.state.get(m.key)
            self.state[m.key] = m
            out += self._closing(m, now, ts)
            if prev is not None and prev.status != m.status:
                out.append(self._transition(prev, m, ts))
            if m.status is MarketStatus.DISPUTED and prev is None:
                out.append(
                    Alert(
                        ts,
                        "disputed",
                        Severity.CRITICAL,
                        m.key,
                        f"market is disputed: {m.question}",
                    )
                )
            if m.status is MarketStatus.RESOLVED and (prev is None or prev.result != m.result):
                out += self._unusual_resolution(m, ts)
            if (
                m.status is MarketStatus.OPEN
                and m.close_time is not None
                and now > m.close_time + timedelta(hours=1)
            ):
                out.append(
                    Alert(
                        ts,
                        "open_past_close",
                        Severity.WARN,
                        m.key,
                        f"still open {now - m.close_time} after scheduled close",
                    )
                )
        return out

    def _closing(self, m: UnifiedMarket, now: datetime, ts: int) -> list[Alert]:
        if m.status is not MarketStatus.OPEN or m.close_time is None or m.key in self.close_warned:
            return []
        left = m.close_time - now
        if timedelta(0) <= left <= self.warn_before:
            self.close_warned.add(m.key)
            return [
                Alert(
                    ts,
                    "closing_soon",
                    Severity.INFO,
                    m.key,
                    f"closes in {left.total_seconds() / 3600:.1f}h: {m.question}",
                    {"close_time": m.close_time.isoformat()},
                )
            ]
        return []

    @staticmethod
    def _transition(prev: UnifiedMarket, m: UnifiedMarket, ts: int) -> Alert:
        sev = Severity.INFO
        if m.status is MarketStatus.DISPUTED:
            sev = Severity.CRITICAL
        elif prev.status is MarketStatus.OPEN and m.status is MarketStatus.RESOLVED:
            sev = Severity.WARN  # skipped the closed/proposed stage
        elif (
            prev.status in (MarketStatus.RESOLVED, MarketStatus.PROPOSED)
            and m.status is MarketStatus.OPEN
        ):
            sev = Severity.CRITICAL  # re-opened
        return Alert(
            ts,
            "status_change",
            sev,
            m.key,
            f"{prev.status.value} -> {m.status.value}: {m.question}",
            {"from": prev.status.value, "to": m.status.value},
        )

    @staticmethod
    def _unusual_resolution(m: UnifiedMarket, ts: int) -> list[Alert]:
        out: list[Alert] = []
        if m.result is not None and m.result not in (0.0, 1.0):
            out.append(
                Alert(
                    ts,
                    "non_binary_resolution",
                    Severity.WARN,
                    m.key,
                    f"resolved at {m.result}: {m.question}",
                    {"result": m.result},
                )
            )
        if "disputed" in m.raw_status:
            out.append(
                Alert(
                    ts,
                    "resolved_after_dispute",
                    Severity.WARN,
                    m.key,
                    f"resolution went through a dispute: {m.raw_status}",
                )
            )
        if (
            m.last_price is not None
            and m.result in (0.0, 1.0)
            and abs(m.last_price - m.result) > 0.5
        ):
            out.append(
                Alert(
                    ts,
                    "surprise_resolution",
                    Severity.INFO,
                    m.key,
                    f"resolved {m.result} vs last price {m.last_price}",
                )
            )
        return out


def unusual_rule_terms(text: str) -> list[str]:
    """Rule wording worth a human look (ties, 50-50, fair price, postponement, void)."""
    return sorted({t.group(0).lower() for t in _UNUSUAL_WORDING.finditer(text or "")})
