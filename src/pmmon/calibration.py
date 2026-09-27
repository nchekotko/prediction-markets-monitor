"""Collect (price H hours before close, resolved outcome) pairs from both venues."""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from pmmon.venues.kalshi import KalshiClient, market_result
from pmmon.venues.polymarket import PolymarketClient, _json_list, parse_ts

log = logging.getLogger(__name__)


@dataclass
class CalibrationPoint:
    venue: str
    market_id: str
    question: str
    close_ts: int
    horizon_h: float
    price: float
    outcome: float
    volume: float


async def kalshi_points(
    client: KalshiClient,
    *,
    horizon_h: float,
    limit: int,
    min_volume: float = 1_000.0,
    min_close_ts: int | None = None,
    series: str | None = None,
) -> list[CalibrationPoint]:
    out: list[CalibrationPoint] = []
    params = {
        "status": "settled",
        "min_close_ts": min_close_ts,
        "series_ticker": series,
        "mve_filter": "exclude",  # multivariate combos flood the settled listing
    }
    async for m in client.iter_markets(max_pages=50, **params):
        if len(out) >= limit:
            break
        tk = m.get("ticker", "")
        if tk.startswith("KXMVE"):  # multivariate combos: parlay-like, skip
            continue
        res = market_result(m)
        vol = float(m.get("volume_fp") or m.get("volume") or 0)
        close = parse_ts(m.get("close_time"))
        if res not in (0.0, 1.0) or vol < min_volume or close is None:
            continue
        end = int(close.timestamp() - horizon_h * 3600)
        series_ticker = m.get("event_ticker", tk).split("-")[0]
        try:
            candles = await client.candlesticks(series_ticker, tk, end - 6 * 3600, end, 60)
        except Exception as exc:
            log.debug("candles %s failed: %s", tk, exc)
            continue
        price = _kalshi_candle_price(candles)
        if price is None:
            continue
        out.append(
            CalibrationPoint(
                "kalshi", tk, m.get("title", ""), int(close.timestamp()), horizon_h, price, res, vol
            )
        )
    return out


def _kalshi_candle_price(candles: list[dict]) -> float | None:
    """Mid of the last candle's closing bid/ask; falls back to last traded price."""
    for c in reversed(candles):
        bid = _num((c.get("yes_bid") or {}).get("close_dollars"))
        ask = _num((c.get("yes_ask") or {}).get("close_dollars"))
        if bid is not None and ask is not None and 0 < bid < ask < 1:
            return (bid + ask) / 2
        px = _num((c.get("price") or {}).get("close_dollars"))
        if px is not None and 0 < px < 1:
            return px
    return None


def _num(v: object) -> float | None:
    try:
        return None if v in (None, "") else float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


async def polymarket_points(
    client: PolymarketClient,
    *,
    horizon_h: float,
    limit: int,
    min_volume: float = 1_000.0,
    end_date_min: str | None = None,
) -> list[CalibrationPoint]:
    out: list[CalibrationPoint] = []
    params = {
        "closed": "true",
        "order": "volumeNum",
        "ascending": "false",
        "end_date_min": end_date_min,
        "volume_num_min": min_volume,
    }
    async for m in client.iter_markets(max_pages=20, **params):
        if len(out) >= limit:
            break
        if (m.get("umaResolutionStatus") or "").lower() != "resolved":
            continue
        prices = [_num(p) for p in _json_list(m.get("outcomePrices"))]
        tokens = _json_list(m.get("clobTokenIds"))
        if len(prices) != 2 or len(tokens) != 2 or prices[0] not in (0.0, 1.0):
            continue  # skip 50-50 / multi-outcome
        close = parse_ts(m.get("closedTime")) or parse_ts(m.get("endDate"))
        if close is None:
            continue
        end = int(close.timestamp() - horizon_h * 3600)
        try:
            hist = await client.price_history(str(tokens[0]), end - 6 * 3600, end, 60)
        except Exception as exc:
            log.debug("history %s failed: %s", m.get("id"), exc)
            continue
        hist = [(t, p) for t, p in hist if t <= end and 0 < p < 1]
        if not hist:
            continue
        out.append(
            CalibrationPoint(
                "polymarket",
                str(tokens[0]),
                m.get("question", ""),
                int(close.timestamp()),
                horizon_h,
                hist[-1][1],
                float(prices[0]),
                float(m.get("volumeNum") or 0),
            )
        )
    return out


FIELDS = list(CalibrationPoint.__dataclass_fields__)


def write_points(path: str | Path, points: list[CalibrationPoint]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for p in points:
            w.writerow(asdict(p))


def read_points(path: str | Path) -> list[CalibrationPoint]:
    with open(path, encoding="utf-8") as fh:
        return [
            CalibrationPoint(
                r["venue"],
                r["market_id"],
                r["question"],
                int(r["close_ts"]),
                float(r["horizon_h"]),
                float(r["price"]),
                float(r["outcome"]),
                float(r["volume"]),
            )
            for r in csv.DictReader(fh)
        ]
