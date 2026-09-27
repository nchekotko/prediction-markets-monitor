from __future__ import annotations

from conftest import REPO, load

from pmmon.mapping.matcher import Overrides, match_markets, resolution_risk
from pmmon.mapping.text import content_tokens, default_aliases, numbers
from pmmon.venues.kalshi import parse_market
from pmmon.venues.polymarket import parse_gamma_market


def _pm(event_file: str):
    ev = load(event_file)[0]
    return [c for m in ev["markets"] for c in parse_gamma_market(m, ev)]


def _kalshi_nfl():
    ev = load("kalshi_event_kcmia.json")["event"]
    series = load("kalshi_series_nflgame.json")["series"]
    return [parse_market(m, ev, series) for m in load("kalshi_markets_kcmia.json")["markets"]]


def _kalshi_fed():
    ev = load("kalshi_event_fed_oct.json")["event"]
    series = load("kalshi_series_fed.json")["series"]
    return [parse_market(m, ev, series) for m in ev["markets"]]


def test_alias_extraction_short_codes_need_uppercase() -> None:
    al = default_aliases()
    ents, _ = al.extract("KC vs MIA (Sep 27)")
    assert ents == {"nfl_kc", "nfl_mia"}
    ents, _ = al.extract("There was no game")  # 'was'/'no' are English words here
    assert ents == set()
    assert al.extract("Los Angeles C vs Buffalo")[0] == {"nfl_lac", "nfl_buf"}


def test_numbers_normalisation() -> None:
    assert numbers("Will BTC be above $100,000 on Oct 1, 2026?") == {"100000", "1"}
    assert numbers("decrease interest rates by 25 bps") == {"25bps"}
    assert numbers("Hike >25bps") == {">25bps"}
    assert numbers("decrease by 50+ bps") == {"50+bps"}
    assert "hold" in content_tokens("Will there be no change in Fed interest rates?")


def test_nfl_moneyline_outcomes_align_across_venues() -> None:
    pm = _pm("pm_event_kcmia.json")  # moneyline + one spread market
    ks = _kalshi_nfl()
    res = match_markets(pm, ks)
    pairs = {(r.a.outcome, r.b.market_id) for r in res}
    assert ("Chiefs", "KXNFLGAME-26SEP27KCMIA-KC") in pairs
    assert ("Dolphins", "KXNFLGAME-26SEP27KCMIA-MIA") in pairs
    for r in res:
        # the moneyline, not the spread market, wins the match
        assert "Spread" not in r.a.question
        assert r.confidence > 0.9
    # the known rule difference between venues is flagged
    flags = ";".join(res[0].risk_flags)
    assert "rule_terms_differ" in flags and "fair price" in flags


def test_fed_buckets_direction_gate_and_overrides() -> None:
    pm = _pm("pm_event_fed_oct.json")
    ks = _kalshi_fed()
    auto = {(r.a.question, r.b.market_id) for r in match_markets(pm, ks)}
    q = {r[0]: r[1] for r in auto}
    assert q["Will the Fed decrease interest rates by 25 bps after the October 2026 meeting?"] == (
        "KXFEDDECISION-26OCT-C25"
    )
    assert q["Will the Fed increase interest rates by 25 bps after the October 2026 meeting?"] == (
        "KXFEDDECISION-26OCT-H25"
    )
    assert q["Will there be no change in Fed interest rates after the October 2026 meeting?"] == (
        "KXFEDDECISION-26OCT-H0"
    )
    # "50+ bps" vs ">25bps": numeric gate refuses the automatic match
    assert not any("50+" in a for a, _ in auto)
    # NO tokens are never mapped
    assert all(r[0] for r in auto)

    ov = Overrides.load(REPO / "config" / "overrides.toml")
    res = match_markets(pm, ks, overrides=ov)
    by_q = {r.a.question: r for r in res}
    r = by_q["Will the Fed increase interest rates by 50+ bps after the October 2026 meeting?"]
    assert r.b.market_id == "KXFEDDECISION-26OCT-H26" and r.source == "override"
    assert len(res) == 5
    # Polymarket gives no resolution-source URL for these markets -> flagged
    assert any("source_unspecified" in f for f in r.risk_flags)


def test_block_override_prevents_match() -> None:
    pm = _pm("pm_event_kcmia.json")
    ks = _kalshi_nfl()
    chiefs = next(c for c in pm if c.outcome == "Chiefs" and "Spread" not in c.question)
    ov = Overrides(block={(chiefs.key, "kalshi:KXNFLGAME-26SEP27KCMIA-KC")})
    res = match_markets(pm, ks, overrides=ov)
    assert all(not (r.a.key == chiefs.key and r.b.market_id.endswith("-KC")) for r in res)


def test_resolution_risk_flags_deadline_and_source() -> None:
    pm = _pm("pm_event_kcmia.json")[0]
    k = _kalshi_nfl()[0]
    flags = resolution_risk(pm, k)
    assert any(f.startswith("close_time_differs") for f in flags)  # Kalshi closes up to 48h later
    assert not any(f.startswith("source_differs") for f in flags)  # both use nfl.com
    changed = k.model_copy(update={"resolution_source": "ESPN (https://www.espn.com)"})
    assert any(f.startswith("source_differs") for f in resolution_risk(pm, changed))
