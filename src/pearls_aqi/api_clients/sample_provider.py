"""Sample-mode providers — deterministic synthetic data, no network.

Selected when ``AQI_PROVIDER=sample`` / ``WEATHER_PROVIDER=sample`` (or as an
automatic fallback in ``--sample`` backfill). Clearly tagged ``data_source =
'sample'`` so synthetic data is never confused with real observations.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from pearls_aqi.data.domain import Location
from pearls_aqi.data.sample import generate_sample_observations
from pearls_aqi.data.schemas import POLLUTANT_COLUMNS, SUBINDEX_COLUMNS, WEATHER_COLUMNS
from pearls_aqi.utils.timeutils import now_utc, utc_floor_hour

_AQ_COLS = [
    "timestamp",
    *POLLUTANT_COLUMNS,
    "aqi",
    "dominant_pollutant",
    *SUBINDEX_COLUMNS,
    "data_source",
]
_WX_COLS = ["timestamp", *WEATHER_COLUMNS, "data_source"]


def _sample_frame(location: Location, start: datetime, end: datetime) -> pd.DataFrame:
    df = generate_sample_observations(location, start, end)
    df["data_source"] = "sample"
    return df


class SampleAirQualityProvider:
    name = "sample"

    def fetch_current(self, location: Location) -> pd.DataFrame:
        now = utc_floor_hour(now_utc())
        df = _sample_frame(location, now - timedelta(hours=2), now)
        return df[[c for c in _AQ_COLS if c in df.columns]].tail(1).reset_index(drop=True)

    def fetch_historical(self, location: Location, start: datetime, end: datetime) -> pd.DataFrame:
        df = _sample_frame(location, start, end)
        return df[[c for c in _AQ_COLS if c in df.columns]].reset_index(drop=True)


class SampleWeatherProvider:
    name = "sample"

    def fetch_current(self, location: Location) -> pd.DataFrame:
        now = utc_floor_hour(now_utc())
        df = _sample_frame(location, now - timedelta(hours=2), now)
        return df[[c for c in _WX_COLS if c in df.columns]].tail(1).reset_index(drop=True)

    def fetch_historical(self, location: Location, start: datetime, end: datetime) -> pd.DataFrame:
        df = _sample_frame(location, start, end)
        return df[[c for c in _WX_COLS if c in df.columns]].reset_index(drop=True)

    def fetch_forecast(self, location: Location, horizon_hours: int) -> pd.DataFrame:
        now = utc_floor_hour(now_utc())
        df = _sample_frame(location, now + timedelta(hours=1), now + timedelta(hours=horizon_hours))
        return df[[c for c in _WX_COLS if c in df.columns]].reset_index(drop=True)
