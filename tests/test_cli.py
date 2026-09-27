"""End-to-end CLI runs on committed sample data (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import REPO, load

from pmmon.cli import main, read_markets
from pmmon.venues.kalshi import parse_market
from pmmon.venues.polymarket import parse_gamma_market

SAMPLE = REPO / "data" / "sample"


def test_map_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ev = load("pm_event_kcmia.json")[0]
    pm = [c for m in ev["markets"] for c in parse_gamma_market(m, ev)]
    kev = load("kalshi_event_kcmia.json")["event"]
    ks = [parse_market(m, kev) for m in load("kalshi_markets_kcmia.json")["markets"]]
    mk = tmp_path / "markets.jsonl"
    mk.write_text("".join(m.model_dump_json() + "\n" for m in pm + ks), encoding="utf-8")
    assert len(read_markets(mk)) == len(pm) + len(ks)
    out = tmp_path / "mapping.json"
    assert main(["map", "--markets", str(mk), "--out", str(out), "--overrides", "none"]) == 0
    mapping = json.loads(out.read_text(encoding="utf-8"))
    assert {(p["a_market"]["outcome"], p["b_market"]["outcome"]) for p in mapping} == {
        ("Chiefs", "Kansas City"),
        ("Dolphins", "Miami"),
    }
    assert "2 pairs" in capsys.readouterr().out


def test_record_from_journal_is_idempotent(tmp_path: Path, capsys) -> None:
    args = [
        "record",
        "--from-journal",
        str(SAMPLE / "journal_kcmia.jsonl.gz"),
        "--out",
        str(tmp_path / "books"),
    ]
    assert main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert first["rows_written"] > 10_000
    assert second["rows_written"] == 0
    assert (
        second["duplicate_rows_skipped"] == first["rows_written"] + first["duplicate_rows_skipped"]
    )


def test_backtest_mm_on_sample(tmp_path: Path, capsys) -> None:
    out = tmp_path / "bt.json"
    rc = main(
        [
            "backtest",
            "--mapping",
            str(SAMPLE / "mapping_kcmia.json"),
            "--store",
            str(SAMPLE / "books"),
            "--strategy",
            "mm",
            "--mm-venue",
            "polymarket",
            "--size",
            "50",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["rows"] > 10_000
    assert res["quotes_placed"] > 0
    assert abs(res["attribution"]["residual_usd"]) < 1e-6
    assert set(res["markout_cents"]) == {"1s", "10s", "60s"}


def test_backtest_arb_on_sample(tmp_path: Path) -> None:
    out = tmp_path / "arb.json"
    rc = main(
        [
            "backtest",
            "--mapping",
            str(SAMPLE / "mapping_kcmia.json"),
            "--store",
            str(SAMPLE / "books"),
            "--strategy",
            "arb",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    res = json.loads(out.read_text())
    assert {"signals", "attempts", "both_legs_filled", "unhedged_contracts"} <= set(res["arb"])


def test_monitor_on_sample(capsys) -> None:
    rc = main(
        [
            "monitor",
            "--mapping",
            str(SAMPLE / "mapping_kcmia.json"),
            "--store",
            str(SAMPLE / "books"),
        ]
    )
    assert rc == 0
    res = json.loads(capsys.readouterr().out)
    assert len(res["pairs"]) == 2
    assert all(p["samples_1s"] > 100 for p in res["pairs"])


def test_calibration_from_csv(capsys) -> None:
    rc = main(["calibration", "--from-csv", str(SAMPLE / "calibration_24h.csv")])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{") :])
    assert payload["all"]["n"] >= 100
    assert 0 < payload["all"]["brier"] < 0.25
