"""Cross-venue market mapping with confidence scores and resolution-risk flags.

Pipeline for every (Polymarket contract, Kalshi contract) candidate pair:

1. **blocking** on dates: the scheduled close / event dates must be within ``max_days``;
2. **entity gate**: canonical entities (teams, assets, institutions) found in both event
   titles must be compatible, numeric lines/strikes must match exactly;
3. **scoring**: fuzzy token-set similarity of event text and of the YES-outcome label,
   entity Jaccard and date proximity -> confidence in [0, 1];
4. **assignment**: greedy one-to-one matching above ``threshold``;
5. **manual overrides** (TOML) force or block specific pairs;
6. **resolution-criteria comparison** -> risk flags (different source, deadline, wording).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from rapidfuzz import fuzz

from pmmon.mapping.text import (
    AliasTable,
    basic_normalise,
    content_tokens,
    default_aliases,
    directions,
    numbers,
)
from pmmon.models import UnifiedMarket
from pmmon.monitor.alerts import unusual_rule_terms

YES_NO = {"yes", "no"}


@dataclass(frozen=True)
class ContractView:
    """Pre-computed matching features of one contract."""

    market: UnifiedMarket
    event_text: str
    outcome_text: str
    entities: frozenset[str]
    outcome_entities: frozenset[str]
    nums: frozenset[str]
    dirs: tuple[frozenset[str], ...]
    date: datetime | None


def _event_date(m: UnifiedMarket) -> datetime | None:
    # Earliest of the known lifecycle times is closest to "when the thing happens".
    times = [t for t in (m.expected_resolution_time, m.close_time) if t is not None]
    return min(times) if times else None


def view(m: UnifiedMarket, aliases: AliasTable | None = None) -> ContractView:
    al = aliases or default_aliases()
    is_yes_no = basic_normalise(m.outcome) in YES_NO
    outcome_text = m.question if is_yes_no else m.outcome
    ev_text = f"{m.event_title} {m.question}"
    ev_ents, _ = al.extract(ev_text)
    out_ents, _ = al.extract(outcome_text)
    return ContractView(
        market=m,
        event_text=" ".join(content_tokens(ev_text, al)),
        outcome_text=" ".join(content_tokens(outcome_text, al)),
        entities=frozenset(ev_ents | out_ents),
        outcome_entities=frozenset(out_ents),
        nums=frozenset(numbers(ev_text) | numbers(outcome_text)),
        dirs=tuple(directions(content_tokens(f"{ev_text} {outcome_text}", al))),
        date=_event_date(m),
    )


@dataclass
class MatchResult:
    a: UnifiedMarket
    b: UnifiedMarket
    confidence: float
    components: dict[str, float]
    risk_flags: list[str] = field(default_factory=list)
    source: str = "auto"  # auto | override

    def as_dict(self) -> dict[str, object]:
        return {
            "a": self.a.key,
            "b": self.b.key,
            "a_label": f"{self.a.event_title} | {self.a.outcome}",
            "b_label": f"{self.b.event_title} | {self.b.outcome}",
            "confidence": round(self.confidence, 3),
            **{k: round(v, 3) for k, v in self.components.items()},
            "risk_flags": ";".join(self.risk_flags),
            "source": self.source,
        }


@dataclass
class Overrides:
    force: list[tuple[str, str]] = field(default_factory=list)
    block: set[tuple[str, str]] = field(default_factory=set)

    @classmethod
    def load(cls, path: str | Path | None) -> Overrides:
        if path is None or not Path(path).exists():
            return cls()
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        force = [(p["a"], p["b"]) for p in data.get("pair", [])]
        block = {(p["a"], p["b"]) for p in data.get("block", [])}
        return cls(force, block)


def _date_score(a: datetime | None, b: datetime | None, max_days: float) -> float | None:
    if a is None or b is None:
        return 0.5
    days = abs((a - b).total_seconds()) / 86400
    if days > max_days:
        return None
    return 1.0 - days / max_days * 0.5 if days > 1 else 1.0


def score_pair(
    a: ContractView, b: ContractView, max_days: float = 3.0
) -> tuple[float, dict[str, float]] | None:
    """Return ``(confidence, components)`` or None when a hard gate rejects the pair."""
    ds = _date_score(a.date, b.date, max_days)
    if ds is None:
        return None
    if a.nums and b.nums and a.nums != b.nums:
        return None
    for da, db in zip(a.dirs, b.dirs, strict=True):
        if da and db and not (da & db):
            return None  # "increase" vs "decrease", "over" vs "under"
    if a.outcome_entities and b.outcome_entities and not (a.outcome_entities & b.outcome_entities):
        return None  # e.g. "Chiefs" vs "Miami" in the same game: different outcomes
    ent_union = a.entities | b.entities
    ent_j = len(a.entities & b.entities) / len(ent_union) if ent_union else 0.5
    if a.entities and b.entities and ent_j == 0:
        return None
    ev = fuzz.token_set_ratio(a.event_text, b.event_text) / 100
    oc = fuzz.token_set_ratio(a.outcome_text, b.outcome_text) / 100
    if a.outcome_entities and a.outcome_entities == b.outcome_entities:
        oc = max(oc, 1.0)
    conf = 0.30 * ev + 0.35 * oc + 0.25 * ent_j + 0.10 * ds
    if bool(a.nums) != bool(b.nums):
        conf *= 0.85  # a line/strike on one side only (e.g. spread vs moneyline)
    return conf, {"event_sim": ev, "outcome_sim": oc, "entity_jaccard": ent_j, "date_score": ds}


def _domain(text: str) -> set[str]:
    out: set[str] = set()
    for tok in text.replace("(", " ").replace(")", " ").replace(";", " ").split():
        if "." in tok:
            host = urlparse(tok if "://" in tok else f"https://{tok}").hostname or ""
            if host and "." in host:
                out.add(host.removeprefix("www."))
    return out


def resolution_risk(a: UnifiedMarket, b: UnifiedMarket, deadline_tol_h: float = 24.0) -> list[str]:
    flags: list[str] = []
    da, db = _domain(a.resolution_source), _domain(b.resolution_source)
    if da and db and not (da & db):
        flags.append(f"source_differs({','.join(sorted(da))}|{','.join(sorted(db))})")
    elif not da or not db:
        flags.append("source_unspecified")
    if a.close_time and b.close_time:
        diff_h = abs((a.close_time - b.close_time).total_seconds()) / 3600
        if diff_h > deadline_tol_h:
            flags.append(f"close_time_differs({diff_h:.0f}h)")
    ta, tb = set(unusual_rule_terms(a.rules_text)), set(unusual_rule_terms(b.rules_text))
    if ta != tb:
        only_a = ",".join(sorted(ta - tb)) or "-"
        only_b = ",".join(sorted(tb - ta)) or "-"
        flags.append(f"rule_terms_differ(a:{only_a}|b:{only_b})")
    if a.tick_size != b.tick_size:
        flags.append(f"tick_differs({a.tick_size}|{b.tick_size})")
    return flags


def match_markets(
    left: list[UnifiedMarket],
    right: list[UnifiedMarket],
    *,
    threshold: float = 0.72,
    max_days: float = 3.0,
    overrides: Overrides | None = None,
    aliases: AliasTable | None = None,
) -> list[MatchResult]:
    ov = overrides or Overrides()
    # For Yes/No markets only the YES token is mapped; NO is its complement.
    lv = [view(m, aliases) for m in left if basic_normalise(m.outcome) != "no"]
    rv = [view(m, aliases) for m in right if basic_normalise(m.outcome) != "no"]
    by_key = {m.key: m for m in left + right}
    results: list[MatchResult] = []
    used_l: set[str] = set()
    used_r: set[str] = set()
    for ka, kb in ov.force:
        if ka in by_key and kb in by_key:
            a, b = by_key[ka], by_key[kb]
            results.append(MatchResult(a, b, 1.0, {}, resolution_risk(a, b), "override"))
            used_l.add(ka)
            used_r.add(kb)
    cands: list[tuple[float, int, int, dict[str, float]]] = []
    for i, a in enumerate(lv):
        if a.market.key in used_l:
            continue
        for j, b in enumerate(rv):
            if b.market.key in used_r or (a.market.key, b.market.key) in ov.block:
                continue
            s = score_pair(a, b, max_days)
            if s is not None and s[0] >= threshold:
                cands.append((s[0], i, j, s[1]))
    cands.sort(key=lambda c: -c[0])
    for conf, i, j, comp in cands:
        a, b = lv[i].market, rv[j].market
        if a.key in used_l or b.key in used_r:
            continue
        used_l.add(a.key)
        used_r.add(b.key)
        results.append(MatchResult(a, b, conf, comp, resolution_risk(a, b)))
    return results
