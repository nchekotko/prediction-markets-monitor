"""Idempotent Parquet storage for normalised book rows.

Layout: ``<root>/venue=<venue>/date=<YYYY-MM-DD>/part-<first_ts>-<last_ts>-<digest>.parquet``
(Hive-style partitions, date = UTC date of the exchange timestamp).

Idempotency: before writing, rows whose key ``(market_id, event_id, kind, side, price)``
already exists in the partition are dropped (keys of existing files are loaded lazily and
cached). The file name is derived from the content digest, so re-ingesting the same journal
is a no-op and a crash mid-flush can simply be retried.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pmmon.models import BookEventKind, BookRow, Side, Venue

SCHEMA = pa.schema(
    [
        ("venue", pa.string()),
        ("market_id", pa.string()),
        ("ts_ms", pa.int64()),
        ("recv_ts_ms", pa.int64()),
        ("seq", pa.int64()),
        ("kind", pa.string()),
        ("side", pa.string()),
        ("price", pa.float64()),
        ("size", pa.float64()),
        ("event_id", pa.string()),
    ]
)

Key = tuple[str, str, str, str, float]


def _key(r: BookRow) -> Key:
    return (r.market_id, r.event_id, r.kind.value, r.side.value, r.price)


def _date(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).strftime("%Y-%m-%d")


class ParquetSink:
    def __init__(self, root: str | Path, flush_rows: int = 50_000) -> None:
        self.root = Path(root)
        self.flush_rows = flush_rows
        self._buf: dict[tuple[str, str], list[BookRow]] = defaultdict(list)
        self._keys: dict[tuple[str, str], set[Key]] = {}
        self.written_rows = 0
        self.duplicate_rows = 0
        self.files_written: list[Path] = []

    def _dir(self, venue: str, date: str) -> Path:
        return self.root / f"venue={venue}" / f"date={date}"

    def _existing_keys(self, part: tuple[str, str]) -> set[Key]:
        if part not in self._keys:
            keys: set[Key] = set()
            d = self._dir(*part)
            if d.exists():
                for f in sorted(d.glob("*.parquet")):
                    t = pq.read_table(f, columns=["market_id", "event_id", "kind", "side", "price"])
                    cols = [t.column(i).to_pylist() for i in range(5)]
                    keys.update(zip(*cols, strict=True))
            self._keys[part] = keys
        return self._keys[part]

    def write(self, rows: Iterable[BookRow]) -> None:
        for r in rows:
            self._buf[(r.venue.value, _date(r.ts_ms))].append(r)
        if sum(len(v) for v in self._buf.values()) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        for part, rows in list(self._buf.items()):
            if not rows:
                continue
            keys = self._existing_keys(part)
            fresh: list[BookRow] = []
            for r in rows:
                k = _key(r)
                if k in keys:
                    self.duplicate_rows += 1
                    continue
                keys.add(k)
                fresh.append(r)
            self._buf[part] = []
            if not fresh:
                continue
            table = rows_to_table(fresh)
            digest = hashlib.sha1(
                "\n".join(repr(_key(r)) for r in fresh).encode(), usedforsecurity=False
            ).hexdigest()[:12]
            d = self._dir(*part)
            d.mkdir(parents=True, exist_ok=True)
            first = min(r.ts_ms for r in fresh)
            last = max(r.ts_ms for r in fresh)
            path = d / f"part-{first}-{last}-{digest}.parquet"
            tmp = path.with_suffix(".tmp")
            pq.write_table(table, tmp, compression="zstd")
            tmp.replace(path)  # atomic rename: readers never see half-written files
            self.files_written.append(path)
            self.written_rows += len(fresh)

    def close(self) -> None:
        self.flush()


def rows_to_table(rows: list[BookRow]) -> pa.Table:
    return pa.table(
        {
            "venue": [r.venue.value for r in rows],
            "market_id": [r.market_id for r in rows],
            "ts_ms": [r.ts_ms for r in rows],
            "recv_ts_ms": [r.recv_ts_ms for r in rows],
            "seq": [r.seq for r in rows],
            "kind": [r.kind.value for r in rows],
            "side": [r.side.value for r in rows],
            "price": [r.price for r in rows],
            "size": [r.size for r in rows],
            "event_id": [r.event_id for r in rows],
        },
        schema=SCHEMA,
    )


def read_rows(
    root: str | Path,
    *,
    venues: Iterable[str] | None = None,
    markets: Iterable[str] | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[BookRow]:
    """Load, dedup and event-time sort rows from the store.

    Sort key: ``(ts_ms, recv_ts_ms, seq, snapshot-before-delta-before-trade)``.
    """
    root = Path(root)
    venue_set = set(venues) if venues else None
    market_set = set(markets) if markets else None
    files = []
    for vdir in sorted(root.glob("venue=*")):
        if venue_set and vdir.name.split("=", 1)[1] not in venue_set:
            continue
        files += sorted(vdir.glob("date=*/*.parquet"))
    seen: set[Key] = set()
    out: list[BookRow] = []
    for f in files:
        t = pq.read_table(f)
        d = t.to_pydict()
        for i in range(t.num_rows):
            m = d["market_id"][i]
            ts = d["ts_ms"][i]
            if market_set and m not in market_set:
                continue
            if (start_ms is not None and ts < start_ms) or (end_ms is not None and ts > end_ms):
                continue
            r = BookRow(
                Venue(d["venue"][i]),
                m,
                ts,
                d["recv_ts_ms"][i],
                d["seq"][i],
                BookEventKind(d["kind"][i]),
                Side(d["side"][i]),
                d["price"][i],
                d["size"][i],
                d["event_id"][i],
            )
            k = _key(r)
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
    order = {BookEventKind.SNAPSHOT: 0, BookEventKind.DELTA: 1, BookEventKind.TRADE: 2}
    out.sort(
        key=lambda r: (r.ts_ms, r.recv_ts_ms, r.venue.value, r.market_id, r.seq, order[r.kind])
    )
    return out


def iter_events(rows: list[BookRow]) -> Iterator[list[BookRow]]:
    """Group consecutive rows of the same (market, event_id, kind) into one event."""
    cur: list[BookRow] = []
    for r in rows:
        if cur and (
            r.market_id != cur[0].market_id
            or r.event_id != cur[0].event_id
            or r.kind != cur[0].kind
            or r.ts_ms != cur[0].ts_ms
        ):
            yield cur
            cur = []
        cur.append(r)
    if cur:
        yield cur
