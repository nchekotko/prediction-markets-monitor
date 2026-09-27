"""Offline monitor: replay recorded books and compute per-pair divergence / arb episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median

from pmmon.book import L2Book
from pmmon.models import BookEventKind, BookRow, FeeModel, Side
from pmmon.monitor.alerts import Alert, BookAlertRules
from pmmon.monitor.metrics import best_arb, mid_divergence
from pmmon.recorder.sink import iter_events


@dataclass
class PairSpec:
    a: str
    b: str
    fees_a: FeeModel
    fees_b: FeeModel
    label: str = ""
    risk_flags: list[str] = field(default_factory=list)


@dataclass
class Episode:
    start_ms: int
    end_ms: int
    max_net_pnl: float
    max_net_edge: float
    direction: str


@dataclass
class PairStats:
    label: str
    samples: int = 0
    divergences: list[float] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)
    _open: Episode | None = None

    def summary(self) -> dict[str, object]:
        absd = [abs(d) for d in self.divergences]
        durs = [(e.end_ms - e.start_ms) / 1000 for e in self.episodes]
        return {
            "pair": self.label,
            "samples_1s": self.samples,
            "mean_abs_mid_divergence_cents": round(mean(absd) * 100, 3) if absd else None,
            "p95_abs_mid_divergence_cents": round(
                sorted(absd)[int(0.95 * (len(absd) - 1))] * 100, 3
            )
            if absd
            else None,
            "arb_episodes": len(self.episodes),
            "median_episode_s": round(median(durs), 2) if durs else None,
            "max_net_pnl_usd": round(max((e.max_net_pnl for e in self.episodes), default=0.0), 3),
            "max_net_edge_cents": round(
                max((e.max_net_edge for e in self.episodes), default=0.0) * 100, 3
            ),
        }


def replay_monitor(
    rows: list[BookRow],
    pairs: list[PairSpec],
    *,
    size: float = 100.0,
    rules: BookAlertRules | None = None,
    sample_ms: int = 1_000,
) -> tuple[list[PairStats], list[Alert]]:
    rules = rules or BookAlertRules()
    books: dict[str, L2Book] = {}
    by_market: dict[str, list[int]] = {}
    for i, p in enumerate(pairs):
        by_market.setdefault(p.a, []).append(i)
        by_market.setdefault(p.b, []).append(i)
    stats = [PairStats(p.label or f"{p.a[:12]}~{p.b[:12]}") for p in pairs]
    alerts: list[Alert] = []
    next_sample = None
    stale_alerted: set[str] = set()
    for ev in iter_events(rows):
        head = ev[0]
        book = books.setdefault(head.market_id, L2Book(head.market_id))
        if head.kind is BookEventKind.SNAPSHOT:
            book.reset(
                [(r.price, r.size) for r in ev if r.side is Side.BID],
                [(r.price, r.size) for r in ev if r.side is Side.ASK],
                head.ts_ms,
            )
        elif head.kind is BookEventKind.DELTA:
            for r in ev:
                book.set_level(r.side, r.price, r.size)
        else:
            continue
        book.last_update_recv_ms = head.ts_ms
        stale_alerted.discard(head.market_id)
        now = head.ts_ms
        # arbitrage episodes: evaluated on every update of a pair member
        for i in by_market.get(head.market_id, []):
            p = pairs[i]
            ba, bb = books.get(p.a), books.get(p.b)
            if ba is None or bb is None or ba.is_crossed or bb.is_crossed:
                continue
            q = best_arb(ba, p.fees_a, bb, p.fees_b, size)
            st = stats[i]
            if q.is_opportunity:
                direction = f"buy {q.buy_market[:10]} / sell {q.sell_market[:10]}"
                if st._open is None:
                    st._open = Episode(now, now, q.net_pnl, q.net_edge, direction)
                    alerts += rules.check_arb(q, now, st.label, p.risk_flags)
                else:
                    st._open.end_ms = now
                    st._open.max_net_pnl = max(st._open.max_net_pnl, q.net_pnl)
                    st._open.max_net_edge = max(st._open.max_net_edge, q.net_edge)
            elif st._open is not None:
                st._open.end_ms = now
                st.episodes.append(st._open)
                st._open = None
        # fixed-cadence sampling of divergence + stale checks
        if next_sample is None:
            next_sample = now
        while now >= next_sample:
            for i, p in enumerate(pairs):
                ba, bb = books.get(p.a), books.get(p.b)
                if ba is None or bb is None:
                    continue
                d = mid_divergence(ba, bb)
                if d is not None:
                    stats[i].divergences.append(d)
                    stats[i].samples += 1
            for m, b in books.items():
                if m in stale_alerted:
                    continue
                for a in rules.check_book(b, next_sample, m):
                    if a.kind == "stale_book":
                        alerts.append(a)
                        stale_alerted.add(m)
            next_sample += sample_ms
    for st in stats:
        if st._open is not None:
            st.episodes.append(st._open)
            st._open = None
    return stats, alerts
