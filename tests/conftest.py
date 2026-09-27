from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).parent.parent


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def pm_event_kcmia() -> dict[str, Any]:
    return load("pm_event_kcmia.json")[0]


@pytest.fixture
def kalshi_kcmia() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    markets = load("kalshi_markets_kcmia.json")["markets"]
    event = load("kalshi_event_kcmia.json")["event"]
    series = load("kalshi_series_nflgame.json")["series"]
    return markets, event, series
