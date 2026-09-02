"""Time handling.

Golden rule: **all timestamps are stored and processed in UTC**. Display-timezone
conversion happens only at the presentation edge (dashboard / API responses),
never inside pipelines or the feature store.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


def now_utc() -> datetime:
    """Timezone-aware current time in UTC."""
    return datetime.now(tz=UTC)


def ensure_utc(value: datetime) -> datetime:
    """Return ``value`` as a timezone-aware UTC datetime.

    Naive datetimes are assumed to already be UTC (a documented convention:
    every provider client localises to UTC before returning).
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def utc_floor_hour(value: datetime | None = None) -> datetime:
    """Floor a datetime to the top of the hour, in UTC."""
    value = ensure_utc(value) if value is not None else now_utc()
    return value.replace(minute=0, second=0, microsecond=0)


def resolve_timezone(name: str) -> ZoneInfo:
    """Resolve an IANA timezone name, raising a clear error on typos."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, KeyError, ValueError) as exc:  # pragma: no cover
        raise ValueError(f"Unknown timezone: {name!r}") from exc


def to_display_tz(value: datetime, tz_name: str) -> datetime:
    """Convert a UTC-aware datetime to a display timezone for presentation."""
    return ensure_utc(value).astimezone(resolve_timezone(tz_name))


def hourly_range(start: datetime, end: datetime, inclusive: bool = True) -> pd.DatetimeIndex:
    """Build an hourly, UTC ``DatetimeIndex`` from ``start`` to ``end``.

    Both endpoints are floored to the hour. Used to validate timestamp
    continuity and to build the forecast horizon index.
    """
    start_h = utc_floor_hour(start)
    end_h = utc_floor_hour(end)
    idx = pd.date_range(start=start_h, end=end_h, freq="h", tz="UTC")
    if not inclusive and len(idx) and idx[-1] == pd.Timestamp(end_h):
        idx = idx[:-1]
    return idx


def to_utc_series(series: pd.Series) -> pd.Series:
    """Coerce a datetime-like Series to UTC-aware timestamps."""
    s = pd.to_datetime(series, utc=True, errors="coerce")
    return s


def parse_iso_or_none(value: str | None) -> datetime | None:
    """Parse an ISO date/datetime string to a UTC-aware datetime, else None."""
    if value is None or value == "":
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return ensure_utc(dt)
