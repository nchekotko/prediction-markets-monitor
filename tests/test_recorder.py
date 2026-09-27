from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FIXTURES, load

from pmmon.book import L2Book
from pmmon.models import BookEventKind, Side, Venue
from pmmon.recorder.journal import JournalWriter, read_journal
from pmmon.recorder.normalise import (
    KalshiRestNormaliser,
    KalshiWsNormaliser,
    PolymarketNormaliser,
)
from pmmon.recorder.recorder import Dispatcher, ingest_journal
from pmmon.recorder.sink import ParquetSink, read_rows

TOK = "tok-yes"


def _book_msg(ts: int, bids, asks, h: str = "h0") -> dict:
    return {
        "event_type": "book",
        "asset_id": TOK,
        "market": "0xabc",
        "timestamp": str(ts),
        "hash": h,
        "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
        "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
    }


def _pc(ts: int, price: float, size: float, side: str, bb: float, ba: float, h: str) -> dict:
    return {
        "event_type": "price_change",
        "market": "0xabc",
        "timestamp": str(ts),
        "price_changes": [
            {
                "asset_id": TOK,
                "price": str(price),
                "size": str(size),
                "side": side,
                "hash": h,
                "best_bid": str(bb),
                "best_ask": str(ba),
            }
        ],
    }


# ------------------------------------------------------------------------------ book


def test_l2book_metrics_and_sweep() -> None:
    b = L2Book("m")
    b.reset([(0.40, 100), (0.39, 200)], [(0.42, 50), (0.43, 300)])
    assert b.best_bid == 0.40 and b.best_ask == 0.42
    assert b.spread == pytest.approx(0.02)
    assert b.mid == pytest.approx(0.41)
    # microprice leans to the thin ask side: (0.40*50 + 0.42*100)/150
    assert b.microprice == pytest.approx((0.40 * 50 + 0.42 * 100) / 150)
    assert b.depth(Side.BID, 0.01) == 300
    filled, vwap = b.sweep(Side.ASK, 100)
    assert filled == 100 and vwap == pytest.approx((50 * 0.42 + 50 * 0.43) / 100)
    b.set_level(Side.ASK, 0.42, 0)
    assert b.best_ask == 0.43


# ------------------------------------------------------------------------------ polymarket


def test_polymarket_book_then_deltas_and_trade() -> None:
    n = PolymarketNormaliser()
    rows = n.on_ws(_book_msg(1000, [(0.40, 100)], [(0.42, 50)]), 1001)
    assert {r.kind for r in rows} == {BookEventKind.SNAPSHOT} and len(rows) == 2
    rows = n.on_ws(_pc(1010, 0.41, 10, "BUY", 0.41, 0.42, "h1"), 1011)
    assert rows[0].kind is BookEventKind.DELTA and rows[0].side is Side.BID
    assert n.books[TOK].best_bid == 0.41
    assert n.stats.gaps == 0
    trade = {
        "event_type": "last_trade_price",
        "asset_id": TOK,
        "price": "0.42",
        "size": "5",
        "side": "BUY",
        "timestamp": "1020",
        "transaction_hash": "0xt",
    }
    rows = n.on_ws(trade, 1021)
    assert rows[0].kind is BookEventKind.TRADE and rows[0].side is Side.ASK  # lifted the ask


def test_polymarket_gap_detected_from_reported_touch_and_resync() -> None:
    n = PolymarketNormaliser()
    n.on_ws(_book_msg(1000, [(0.40, 100)], [(0.42, 50)]), 1001)
    # A message that would have removed the 0.42 ask was lost: venue reports best_ask 0.43
    n.on_ws(_pc(1010, 0.43, 70, "SELL", 0.40, 0.43, "h2"), 1011)
    assert n.stats.gaps == 1 and TOK in n.needs_resync
    # REST snapshot clears the flag
    n.on_book({**_book_msg(1020, [(0.40, 100)], [(0.43, 70)]), "hash": "rest"}, 1021)
    assert TOK not in n.needs_resync and n.stats.resyncs == 1


def test_polymarket_delta_before_snapshot_is_dropped_and_flags_resync() -> None:
    n = PolymarketNormaliser()
    assert n.on_ws(_pc(1010, 0.41, 10, "BUY", 0.41, 0.42, "h1"), 1011) == []
    assert n.stats.dropped_before_snapshot == 1 and TOK in n.needs_resync


def test_synthetic_snapshot_every_interval() -> None:
    n = PolymarketNormaliser(snapshot_interval_ms=1000)
    n.on_ws(_book_msg(0, [(0.40, 100)], [(0.42, 50)]), 0)
    rows = n.on_ws(_pc(1500, 0.41, 10, "BUY", 0.41, 0.42, "h1"), 1500)
    kinds = [r.kind for r in rows]
    assert BookEventKind.SNAPSHOT in kinds and rows[-1].event_id == "synth:1500"


def test_real_polymarket_ws_sample_replays_consistently() -> None:
    """Recorded market-channel messages (Chiefs vs Dolphins, 2026-09-27)."""
    n = PolymarketNormaliser()
    total = 0
    for rec in read_journal(FIXTURES / "pm_ws_sample.jsonl"):
        total += len(n.on_ws(rec.payload, rec.recv_ts_ms))
    assert total > 100
    assert n.stats.messages > 50
    for book in n.books.values():
        assert book.best_bid is not None and book.best_ask is not None
    # gaps are rare and always repaired by the venue's next `book` message
    assert n.stats.gaps <= n.stats.messages * 0.05


# ------------------------------------------------------------------------------ kalshi


def test_kalshi_rest_diff_produces_absolute_deltas() -> None:
    n = KalshiRestNormaliser()
    ob = load("kalshi_orderbook_kc.json")
    first = n.on_rest_book("KC", ob, 1000)
    assert all(r.kind is BookEventKind.SNAPSHOT for r in first)
    assert n.on_rest_book("KC", ob, 2000) == []  # unchanged book -> nothing
    changed = {
        "orderbook_fp": {
            "yes_dollars": ob["orderbook_fp"]["yes_dollars"][:-1],  # best YES bid pulled
            "no_dollars": ob["orderbook_fp"]["no_dollars"],
        }
    }
    rows = n.on_rest_book("KC", changed, 3000)
    assert len(rows) == 1 and rows[0].size == 0 and rows[0].side is Side.BID


def test_kalshi_ws_seq_gap() -> None:
    n = KalshiWsNormaliser()
    snap = {
        "type": "orderbook_snapshot",
        "sid": 2,
        "seq": 1,
        "msg": {
            "market_ticker": "FED",
            "yes_dollars_fp": [["0.08", "300"]],
            "no_dollars_fp": [["0.54", "20"]],
        },
    }
    n.on_ws(snap, 1)
    assert n.books["FED"].best_ask == pytest.approx(0.46)
    d = {
        "type": "orderbook_delta",
        "sid": 2,
        "seq": 2,
        "msg": {
            "market_ticker": "FED",
            "price_dollars": "0.08",
            "delta_fp": "-100",
            "side": "yes",
            "ts_ms": 5,
        },
    }
    rows = n.on_ws(d, 5)
    assert rows[0].size == pytest.approx(200)
    n.on_ws({**d, "seq": 4}, 6)  # seq 3 missing
    assert n.stats.gaps == 1 and "FED" in n.needs_resync
    n.on_ws({**snap, "seq": 5}, 7)
    assert "FED" not in n.needs_resync


# ------------------------------------------------------------------------------ sink / journal


def test_parquet_sink_is_idempotent(tmp_path: Path) -> None:
    n = PolymarketNormaliser()
    rows = n.on_ws(_book_msg(1_790_000_000_000, [(0.40, 100)], [(0.42, 50)]), 1)
    rows += n.on_ws(_pc(1_790_000_000_500, 0.41, 10, "BUY", 0.41, 0.42, "h1"), 2)
    s1 = ParquetSink(tmp_path)
    s1.write(rows)
    s1.close()
    s2 = ParquetSink(tmp_path)  # new process: keys loaded from disk
    s2.write(rows)
    s2.close()
    assert s1.written_rows == 3 and s2.written_rows == 0 and s2.duplicate_rows == 3
    files = list(tmp_path.rglob("*.parquet"))
    assert (
        len(files) == 1 and "venue=polymarket" in str(files[0]) and "date=2026-09" in str(files[0])
    )
    back = read_rows(tmp_path)
    assert [r.kind for r in back] == [
        BookEventKind.SNAPSHOT,
        BookEventKind.SNAPSHOT,
        BookEventKind.DELTA,
    ]


def test_journal_roundtrip_and_ingest_twice(tmp_path: Path) -> None:
    j = JournalWriter(tmp_path / "j.jsonl.gz")
    j.write("polymarket", 1, "ws", _book_msg(1_790_000_000_000, [(0.4, 1)], [(0.5, 1)]))
    j.write("kalshi", 2, "rest_book", {"ticker": "KC", **load("kalshi_orderbook_kc.json")})
    j.close()
    assert [r.kind for r in read_journal(tmp_path / "j.jsonl.gz")] == ["ws", "rest_book"]
    d1 = ingest_journal(tmp_path / "j.jsonl.gz", ParquetSink(tmp_path / "store"))
    d2 = ingest_journal(tmp_path / "j.jsonl.gz", ParquetSink(tmp_path / "store"))
    assert d1.sink.written_rows > 0 and d2.sink.written_rows == 0
    venues = {r.venue for r in read_rows(tmp_path / "store")}
    assert venues == {Venue.POLYMARKET, Venue.KALSHI}


def test_truncated_journal_line_is_skipped(tmp_path: Path) -> None:
    p = tmp_path / "j.jsonl"
    p.write_text(
        '{"venue":"kalshi","recv_ts_ms":1,"kind":"x","payload":{}}\n{"venue":"kal', encoding="utf-8"
    )
    assert len(list(read_journal(p))) == 1


def test_dispatcher_reports_pending_resyncs(tmp_path: Path) -> None:
    d = Dispatcher(sink=ParquetSink(tmp_path))
    d.dispatch("polymarket", 1, "ws", _pc(10, 0.41, 10, "BUY", 0.41, 0.42, "h"))
    assert d.pending_resyncs()["polymarket"] == {TOK}
    d.dispatch("polymarket", 2, "rest_book", _book_msg(11, [(0.41, 10)], [(0.42, 5)]))
    assert d.pending_resyncs()["polymarket"] == set()
