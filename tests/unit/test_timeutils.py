"""Unit tests for time utilities (UTC-internal, display conversion)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from pearls_aqi.utils.timeutils import (
    ensure_utc,
    hourly_range,
    to_display_tz,
    utc_floor_hour,
)


def test_ensure_utc_naive_assumed_utc():
    dt = datetime(2025, 1, 1, 12, 0, 0)
    out = ensure_utc(dt)
    assert out.tzinfo == UTC
    assert out.hour == 12


def test_ensure_utc_converts_aware():
    dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    out = ensure_utc(dt)
    assert out == dt


def test_utc_floor_hour():
    dt = datetime(2025, 1, 1, 12, 45, 30, tzinfo=UTC)
    assert utc_floor_hour(dt) == datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


def test_to_display_tz_roundtrip():
    dt = datetime(2025, 6, 1, 6, 0, 0, tzinfo=UTC)
    karachi = to_display_tz(dt, "Asia/Karachi")  # UTC+5
    assert karachi.hour == 11
    # Converting back to UTC recovers the original instant.
    assert ensure_utc(karachi) == dt


def test_hourly_range_continuity():
    start = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    end = datetime(2025, 1, 1, 5, 0, tzinfo=UTC)
    idx = hourly_range(start, end)
    assert isinstance(idx, pd.DatetimeIndex)
    assert len(idx) == 6
    assert (idx.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
