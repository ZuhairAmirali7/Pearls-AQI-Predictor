"""Contract tests: provider outputs match the canonical schema."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from pearls_aqi.api_clients.sample_provider import (
    SampleAirQualityProvider,
    SampleWeatherProvider,
)
from pearls_aqi.data.domain import Location
from pearls_aqi.data.schemas import POLLUTANT_COLUMNS, WEATHER_COLUMNS

pytestmark = pytest.mark.contract

LOC = Location(
    city="Karachi", country="Pakistan", latitude=24.86, longitude=67.0, timezone="Asia/Karachi"
)


def test_air_quality_provider_contract():
    end = datetime.now(tz=UTC)
    df = SampleAirQualityProvider().fetch_historical(LOC, end - timedelta(days=3), end)
    assert "timestamp" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    for col in ["pm2_5", "pm10", "o3", "no2", "so2", "co"]:
        assert col in df.columns
    assert set(df.columns) & set(POLLUTANT_COLUMNS)


def test_weather_provider_forecast_contract():
    df = SampleWeatherProvider().fetch_forecast(LOC, horizon_hours=24)
    assert "timestamp" in df.columns
    assert len(df) == 24
    for col in ["temperature", "humidity", "wind_speed", "pressure"]:
        assert col in df.columns
    assert set(df.columns) & set(WEATHER_COLUMNS)


def test_timestamps_are_utc():
    end = datetime.now(tz=UTC)
    df = SampleAirQualityProvider().fetch_historical(LOC, end - timedelta(days=1), end)
    assert str(df["timestamp"].dt.tz) == "UTC"
