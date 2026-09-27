"""Command-line interface: discover, map, record, monitor, backtest, calibration."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pmmon.models import FeeModel, UnifiedMarket, Venue

log = logging.getLogger("pmmon")


# ---------------------------------------------------------------------------- helpers
def _write_markets(path: Path, markets: list[UnifiedMarket]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for m in markets:
            fh.write(m.model_dump_json() + "\n")


def read_markets(path: str | Path) -> list[UnifiedMarket]:
    with open(path, encoding="utf-8") as fh:
        return [UnifiedMarket.model_validate_json(line) for line in fh if line.strip()]


def read_mapping(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _print_json(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------- discover
async def discover(args: argparse.Namespace) -> list[UnifiedMarket]:
    from pmmon.venues.kalshi import KalshiClient, parse_market
    from pmmon.venues.polymarket import PolymarketClient, parse_gamma_market

    out: list[UnifiedMarket] = []
    q = (args.query or "").lower()
    if args.venue in ("polymarket", "both"):
        async with PolymarketClient() as pm:
            params: dict[str, Any] = {
                "closed": "false",
                "order": "volume24hr",
                "ascending": "false",
                "limit": min(args.limit, 100),
            }
            if args.pm_tag:
                params["tag_slug"] = args.pm_tag
            events = await pm.events(**params)
            for ev in events:
                for m in ev.get("markets", []):
                    if args.moneyline_only and m.get("sportsMarketType") not in (None, "moneyline"):
                        continue
                    if m.get("closed"):
                        continue
                    for um in parse_gamma_market(m, ev):
                        if q in f"{um.event_title} {um.question}".lower():
                            out.append(um)
    if args.venue in ("kalshi", "both"):
        async with KalshiClient() as k:
            n = 0
            async for ev in k.iter_events(
                max_pages=max(1, args.limit // 100),
                page_size=100,
                status="open",
                with_nested_markets="true",
                series_ticker=args.kalshi_series,
            ):
                series = await k.series(ev["series_ticker"]) if ev.get("series_ticker") else None
                for m in ev.get("markets", []):
                    um = parse_market(m, ev, series)
                    if q in f"{um.event_title} {um.question}".lower():
                        out.append(um)
                n += 1
                if n >= args.limit:
                    break
    return out


def cmd_discover(args: argparse.Namespace) -> int:
    markets = asyncio.run(discover(args))
    _write_markets(Path(args.out), markets)
    by_venue: dict[str, int] = {}
    for m in markets:
        by_venue[m.venue.value] = by_venue.get(m.venue.value, 0) + 1
    _print_json({"written": str(args.out), "contracts": by_venue})
    return 0


# ---------------------------------------------------------------------------- map
def cmd_map(args: argparse.Namespace) -> int:
    from pmmon.mapping.matcher import Overrides, match_markets

    markets = read_markets(args.markets)
    left = [m for m in markets if m.venue is Venue.POLYMARKET]
    right = [m for m in markets if m.venue is Venue.KALSHI]
    res = match_markets(
        left,
        right,
        threshold=args.threshold,
        max_days=args.max_days,
        overrides=Overrides.load(args.overrides),
    )
    res.sort(key=lambda r: -r.confidence)
    payload = [
        {
            **r.as_dict(),
            "risk_flags": r.risk_flags,
            "a_market": r.a.model_dump(mode="json"),
            "b_market": r.b.model_dump(mode="json"),
        }
        for r in res
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    for r in res:
        d = r.as_dict()
        sys.stdout.write(
            f"{d['confidence']:.3f}  {d['a_label'][:45]:45s} <-> {d['b_label'][:45]:45s}"
            f"  {d['risk_flags'][:80]}\n"
        )
    sys.stdout.write(f"{len(res)} pairs -> {args.out}\n")
    return 0


# ---------------------------------------------------------------------------- record
def _mapping_targets(path: str | None) -> tuple[list[str], list[str]]:
    if not path:
        return [], []
    pm, ks = [], []
    for p in read_mapping(path):
        for side in ("a_market", "b_market"):
            m = UnifiedMarket.model_validate(p[side])
            (pm if m.venue is Venue.POLYMARKET else ks).append(m.market_id)
    return sorted(set(pm)), sorted(set(ks))


def cmd_record(args: argparse.Namespace) -> int:
    from pmmon.recorder.recorder import LiveRecorder, ingest_journal
    from pmmon.recorder.sink import ParquetSink

    if args.from_journal:
        d = ingest_journal(args.from_journal, ParquetSink(args.out))
        _print_json(d.stats())
        return 0
    pm, ks = _mapping_targets(args.mapping)
    pm += args.pm_token or []
    ks += args.kalshi_ticker or []
    if not pm and not ks:
        sys.stderr.write("nothing to record: pass --mapping or --pm-token/--kalshi-ticker\n")
        return 2
    rec = LiveRecorder(
        out_dir=args.out,
        pm_tokens=sorted(set(pm)),
        kalshi_tickers=sorted(set(ks)),
        journal_path=args.journal,
        kalshi_poll_s=args.kalshi_poll_s,
    )
    stats = asyncio.run(rec.run(args.duration))
    _print_json(stats)
    return 0


# ---------------------------------------------------------------------------- monitor
def _pairs_from_mapping(path: str) -> list[Any]:
    from pmmon.monitor.replay import PairSpec

    specs = []
    for p in read_mapping(path):
        a = UnifiedMarket.model_validate(p["a_market"])
        b = UnifiedMarket.model_validate(p["b_market"])
        specs.append(
            PairSpec(
                a.market_id,
                b.market_id,
                a.fees,
                b.fees,
                label=f"{a.event_title} | {a.outcome} ~ {b.outcome}",
                risk_flags=list(p.get("risk_flags") or []),
            )
        )
    return specs


def cmd_monitor(args: argparse.Namespace) -> int:
    from pmmon.monitor.alerts import BookAlertRules
    from pmmon.monitor.replay import replay_monitor
    from pmmon.recorder.sink import read_rows

    if args.resolution:
        return asyncio.run(_resolution_once(args))
    pairs = _pairs_from_mapping(args.mapping)
    markets = {p.a for p in pairs} | {p.b for p in pairs}
    rows = read_rows(args.store, markets=markets)
    rules = BookAlertRules(stale_after_ms=int(args.stale_s * 1000), min_arb_net_usd=args.min_net)
    stats, alerts = replay_monitor(rows, pairs, size=args.size, rules=rules)
    counts: dict[str, int] = {}
    for a in alerts:
        counts[a.kind] = counts.get(a.kind, 0) + 1
    _print_json(
        {
            "rows": len(rows),
            "pairs": [s.summary() for s in stats],
            "alert_counts": counts,
            "first_alerts": [a.as_dict() for a in alerts[: args.show]],
        }
    )
    if args.alerts_out:
        with open(args.alerts_out, "w", encoding="utf-8") as fh:
            for a in alerts:
                fh.write(json.dumps(a.as_dict(), ensure_ascii=False) + "\n")
    return 0


async def _resolution_once(args: argparse.Namespace) -> int:
    """Re-fetch metadata for mapped markets and diff against the mapping snapshot."""
    from datetime import timedelta

    from pmmon.monitor.alerts import ResolutionMonitor
    from pmmon.venues.kalshi import KalshiClient
    from pmmon.venues.polymarket import PolymarketClient, parse_gamma_market

    old: list[UnifiedMarket] = []
    for p in read_mapping(args.mapping):
        old += [
            UnifiedMarket.model_validate(p["a_market"]),
            UnifiedMarket.model_validate(p["b_market"]),
        ]
    mon = ResolutionMonitor(warn_before=timedelta(hours=args.warn_hours))
    now = datetime.now(UTC)
    mon.update(old, now)
    mon.close_warned.clear()
    fresh: list[UnifiedMarket] = []
    async with PolymarketClient() as pm, KalshiClient() as k:
        pm_parents = {m.parent_id for m in old if m.venue is Venue.POLYMARKET}
        for cid in sorted(pm_parents):
            res = await pm.gamma.get_json("/markets", {"condition_ids": cid})
            for m in res:
                fresh += parse_gamma_market(m)
        for m in old:
            if m.venue is Venue.KALSHI:
                raw = await k.rest.get_json(f"/markets/{m.market_id}")
                fresh.append(await k.unified(raw["market"]))
    alerts = mon.update(fresh, now)
    _print_json([a.as_dict() for a in alerts])
    return 0


# ---------------------------------------------------------------------------- backtest
def cmd_backtest(args: argparse.Namespace) -> int:
    from pmmon.backtest.engine import BacktestConfig, Backtester
    from pmmon.backtest.strategies import ArbPair, CrossVenueArb, PassiveMarketMaker
    from pmmon.recorder.sink import read_rows

    mapping = read_mapping(args.mapping)
    fees: dict[str, FeeModel] = {}
    ticks: dict[str, float] = {}
    pairs: list[ArbPair] = []
    for p in mapping:
        a = UnifiedMarket.model_validate(p["a_market"])
        b = UnifiedMarket.model_validate(p["b_market"])
        for m in (a, b):
            fees[m.market_id] = m.fees
            ticks[m.market_id] = m.tick_size
        pairs.append(ArbPair(a.market_id, b.market_id, list(p.get("risk_flags") or [])))
    markets = set(fees)
    if args.markets:
        markets &= set(args.markets)
    cfg = BacktestConfig(
        latency_ms={Venue.POLYMARKET: args.pm_latency_ms, Venue.KALSHI: args.kalshi_latency_ms},
        ioc_confirm_ms={Venue.POLYMARKET: 0, Venue.KALSHI: args.kalshi_confirm_ms},
    )
    rows = read_rows(args.store, markets=markets)
    if args.strategy == "mm":
        venue_filter = {v for v in (args.mm_venue or [])}
        mm_markets = sorted(
            {r.market_id for r in rows if not venue_filter or r.venue.value in venue_filter}
        )
        strat: Any = PassiveMarketMaker(
            markets=mm_markets,
            tick=ticks,
            size=args.size,
            half_spread=args.half_spread,
            max_inventory=args.max_inventory,
            skew=args.skew,
        )
    else:
        strat = CrossVenueArb(pairs=pairs, size=args.size, min_edge=args.min_edge)
    res = Backtester(rows, fees, strat, cfg).run()
    out = {"strategy": args.strategy, "rows": len(rows), **res.summary()}
    if isinstance(strat, CrossVenueArb):
        out["arb"] = strat.leg_stats(res.fills)
    _print_json(out)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------- calibration
async def _collect_calibration(args: argparse.Namespace) -> list[Any]:
    from pmmon.calibration import kalshi_points, polymarket_points
    from pmmon.venues.kalshi import KalshiClient
    from pmmon.venues.polymarket import PolymarketClient

    pts: list[Any] = []
    if args.venue in ("polymarket", "both"):
        async with PolymarketClient() as pm:
            pts += await polymarket_points(
                pm,
                horizon_h=args.horizon_h,
                limit=args.limit,
                min_volume=args.min_volume,
                end_date_min=args.since,
            )
    if args.venue in ("kalshi", "both"):
        since = None
        if args.since:
            since = int(datetime.fromisoformat(args.since).replace(tzinfo=UTC).timestamp())
        async with KalshiClient() as k:
            pts += await kalshi_points(
                k,
                horizon_h=args.horizon_h,
                limit=args.limit,
                min_volume=args.min_volume,
                min_close_ts=since,
                series=args.kalshi_series,
            )
    return pts


def cmd_calibration(args: argparse.Namespace) -> int:
    from pmmon.bayes import calibration_report
    from pmmon.calibration import read_points, write_points

    if args.from_csv:
        pts = read_points(args.from_csv)
    else:
        pts = asyncio.run(_collect_calibration(args))
        if args.csv:
            write_points(args.csv, pts)
    if not pts:
        sys.stderr.write("no calibration points\n")
        return 1
    out: dict[str, Any] = {}
    groups: dict[str, list[Any]] = {"all": pts}
    for p in pts:
        groups.setdefault(p.venue, []).append(p)
    for name, g in groups.items():
        rep = calibration_report([p.price for p in g], [p.outcome for p in g], n_bins=args.bins)
        out[name] = rep.summary()
        if name == "all" or args.tables:
            sys.stdout.write(f"\n[{name}] reliability (| = mean prediction, # = observed)\n")
            sys.stdout.write(rep.table() + "\n")
    _print_json(out)
    return 0


# ---------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pmmon", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="fetch open markets into unified JSONL")
    d.add_argument("--venue", choices=["polymarket", "kalshi", "both"], default="both")
    d.add_argument("--query", default="")
    d.add_argument("--pm-tag", default=None, help="Polymarket tag slug, e.g. nfl")
    d.add_argument("--kalshi-series", default=None, help="Kalshi series ticker, e.g. KXNFLGAME")
    d.add_argument("--moneyline-only", action="store_true")
    d.add_argument("--limit", type=int, default=100, help="max events per venue")
    d.add_argument("--out", default="data/markets.jsonl")
    d.set_defaults(func=cmd_discover)

    m = sub.add_parser("map", help="match equivalent contracts across venues")
    m.add_argument("--markets", default="data/markets.jsonl")
    m.add_argument("--overrides", default="config/overrides.toml")
    m.add_argument("--threshold", type=float, default=0.72)
    m.add_argument("--max-days", type=float, default=3.0)
    m.add_argument("--out", default="data/mapping.json")
    m.set_defaults(func=cmd_map)

    r = sub.add_parser("record", help="record L2 books to Parquet (or ingest a journal)")
    r.add_argument("--mapping", default=None)
    r.add_argument("--pm-token", action="append")
    r.add_argument("--kalshi-ticker", action="append")
    r.add_argument("--duration", type=float, default=600.0, help="seconds")
    r.add_argument("--kalshi-poll-s", type=float, default=2.0)
    r.add_argument("--journal", default=None, help="raw JSONL journal path (.gz allowed)")
    r.add_argument("--from-journal", default=None, help="rebuild the store from a journal")
    r.add_argument("--out", default="data/books")
    r.set_defaults(func=cmd_record)

    mo = sub.add_parser("monitor", help="spread/depth/divergence/arb/stale alerts; resolution")
    mo.add_argument("--mapping", default="data/mapping.json")
    mo.add_argument("--store", default="data/books")
    mo.add_argument("--size", type=float, default=100.0, help="contracts per arb check")
    mo.add_argument("--min-net", type=float, default=0.5, help="min net USD for arb alert")
    mo.add_argument("--stale-s", type=float, default=60.0)
    mo.add_argument("--show", type=int, default=10)
    mo.add_argument("--alerts-out", default=None)
    mo.add_argument("--resolution", action="store_true", help="live resolution-status check")
    mo.add_argument("--warn-hours", type=float, default=6.0)
    mo.set_defaults(func=cmd_monitor)

    b = sub.add_parser("backtest", help="event-time backtest over the Parquet store")
    b.add_argument("--mapping", default="data/mapping.json")
    b.add_argument("--store", default="data/books")
    b.add_argument("--strategy", choices=["mm", "arb"], default="mm")
    b.add_argument("--markets", nargs="*", default=None)
    b.add_argument("--mm-venue", action="append", choices=["polymarket", "kalshi"])
    b.add_argument("--size", type=float, default=100.0)
    b.add_argument("--half-spread", type=float, default=0.005)
    b.add_argument("--max-inventory", type=float, default=1000.0)
    b.add_argument("--skew", type=float, default=0.02)
    b.add_argument("--min-edge", type=float, default=0.005)
    b.add_argument("--pm-latency-ms", type=int, default=250)
    b.add_argument("--kalshi-latency-ms", type=int, default=150)
    b.add_argument(
        "--kalshi-confirm-ms",
        type=int,
        default=3000,
        help="IOC on REST-polled Kalshi books fills only liquidity that survives this window",
    )
    b.add_argument("--out", default=None)
    b.set_defaults(func=cmd_backtest)

    c = sub.add_parser("calibration", help="Brier / reliability of prices vs resolved outcomes")
    c.add_argument("--venue", choices=["polymarket", "kalshi", "both"], default="both")
    c.add_argument("--horizon-h", type=float, default=24.0)
    c.add_argument("--limit", type=int, default=200, help="max markets per venue")
    c.add_argument("--min-volume", type=float, default=10_000.0)
    c.add_argument("--since", default=None, help="ISO date: only markets closing after it")
    c.add_argument("--kalshi-series", default=None)
    c.add_argument("--bins", type=int, default=10)
    c.add_argument("--csv", default=None, help="save collected points")
    c.add_argument("--from-csv", default=None, help="offline: load points from CSV")
    c.add_argument("--tables", action="store_true", help="print per-venue tables too")
    c.set_defaults(func=cmd_calibration)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
