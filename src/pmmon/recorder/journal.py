"""Raw message journal (JSON Lines, optionally gzip).

Every inbound message is appended *before* normalisation as
``{"venue", "recv_ts_ms", "kind", "payload"}``. The journal is the source of truth:
the Parquet store can always be rebuilt from it with ``pmmon record --from-journal``,
and because the sink is idempotent, replaying a journal twice changes nothing.

Kinds: ``ws`` (Polymarket market channel), ``rest_book`` (REST snapshot, both venues),
``kalshi_ws`` (authenticated Kalshi WS), ``gamma_market`` / ``market`` (metadata).
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any


@dataclass(frozen=True, slots=True)
class JournalRecord:
    venue: str
    recv_ts_ms: int
    kind: str
    payload: Any


def _open(path: Path, mode: str) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")  # type: ignore[return-value]
    return open(path, mode, encoding="utf-8", buffering=1)


class JournalWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = _open(self.path, "a")
        self.count = 0

    def write(self, venue: str, recv_ts_ms: int, kind: str, payload: Any) -> None:
        self._fh.write(
            json.dumps(
                {"venue": venue, "recv_ts_ms": recv_ts_ms, "kind": kind, "payload": payload},
                separators=(",", ":"),
            )
            + "\n"
        )
        self.count += 1

    def close(self) -> None:
        self._fh.close()


def read_journal(path: str | Path) -> Iterator[JournalRecord]:
    """Yield records; a truncated trailing line (crash mid-write) is skipped."""
    with _open(Path(path), "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield JournalRecord(d["venue"], int(d["recv_ts_ms"]), d["kind"], d["payload"])
