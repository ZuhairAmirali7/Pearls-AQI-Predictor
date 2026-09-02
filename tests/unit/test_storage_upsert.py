"""Unit tests for the idempotent upsert helper."""

from __future__ import annotations

import pandas as pd

from pearls_aqi.storage.base import upsert


def _row(city, ts, val):
    return {"city_id": city, "timestamp": pd.Timestamp(ts, tz="UTC"), "value": val}


def test_upsert_dedups_keeping_last():
    existing = pd.DataFrame([_row("c", "2025-01-01T00:00", 1), _row("c", "2025-01-01T01:00", 2)])
    new = pd.DataFrame([_row("c", "2025-01-01T01:00", 99), _row("c", "2025-01-01T02:00", 3)])
    merged = upsert(existing, new)
    assert len(merged) == 3  # one overlapping key deduped
    # Latest write wins for the overlapping timestamp.
    overlap = merged[merged["timestamp"] == pd.Timestamp("2025-01-01T01:00", tz="UTC")]
    assert overlap["value"].iloc[0] == 99


def test_upsert_sorts_by_key():
    df = pd.DataFrame([_row("c", "2025-01-01T02:00", 3), _row("c", "2025-01-01T00:00", 1)])
    merged = upsert(pd.DataFrame(), df)
    assert list(merged["timestamp"]) == sorted(merged["timestamp"])


def test_upsert_handles_empty_inputs():
    df = pd.DataFrame([_row("c", "2025-01-01T00:00", 1)])
    assert len(upsert(pd.DataFrame(), df)) == 1
    assert len(upsert(df, pd.DataFrame())) == 1
