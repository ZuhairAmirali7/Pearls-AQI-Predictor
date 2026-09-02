"""Unit tests for data-provider parsing and HTTP resilience (mocked, no network)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from pearls_aqi.api_clients.base import BaseHTTPClient
from pearls_aqi.api_clients.open_meteo import (
    OpenMeteoAirQualityProvider,
    OpenMeteoWeatherProvider,
)
from pearls_aqi.api_clients.openweather import OpenWeatherAirQualityProvider
from pearls_aqi.data.domain import Location
from pearls_aqi.exceptions import ProviderResponseError

LOC = Location(
    city="Karachi", country="Pakistan", latitude=24.86, longitude=67.0, timezone="Asia/Karachi"
)
START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 2, tzinfo=UTC)


class FakeClient:
    """Stub HTTP client returning a preset JSON payload."""

    def __init__(self, payload):
        self.payload = payload

    def get_json(self, url, params):
        return self.payload


def test_openmeteo_air_quality_parsing():
    payload = {
        "hourly": {
            "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
            "pm2_5": [10, 12],
            "pm10": [20, 22],
            "carbon_monoxide": [200, 210],
            "nitrogen_monoxide": [1, 1],
            "nitrogen_dioxide": [15, 16],
            "ozone": [30, 31],
            "sulphur_dioxide": [5, 5],
            "ammonia": [2, 2],
        }
    }
    provider = OpenMeteoAirQualityProvider(client=FakeClient(payload))
    df = provider.fetch_historical(LOC, START, END)
    assert {"timestamp", "pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3"} <= set(df.columns)
    assert len(df) == 2
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["pm2_5"].iloc[0] == 10


def test_openmeteo_weather_parsing_and_wmo_mapping():
    payload = {
        "hourly": {
            "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
            "temperature_2m": [20.0, 21.0],
            "apparent_temperature": [19.0, 20.0],
            "relative_humidity_2m": [60, 62],
            "surface_pressure": [1010, 1011],
            "wind_speed_10m": [3.0, 3.5],
            "wind_direction_10m": [180, 200],
            "wind_gusts_10m": [5.0, 6.0],
            "precipitation": [0.0, 0.2],
            "rain": [0.0, 0.2],
            "cloud_cover": [10, 80],
            "visibility": [10000, 8000],
            "dew_point_2m": [12.0, 13.0],
            "weather_code": [0, 61],
        }
    }
    provider = OpenMeteoWeatherProvider(client=FakeClient(payload))
    df = provider.fetch_historical(LOC, START, END)
    assert "temperature" in df.columns and "weather_condition" in df.columns
    assert df["weather_condition"].iloc[0] == "Clear"
    assert df["weather_condition"].iloc[1] == "Rain"


def test_openmeteo_empty_payload_raises():
    provider = OpenMeteoAirQualityProvider(client=FakeClient({}))
    with pytest.raises(ProviderResponseError):
        provider.fetch_historical(LOC, START, END)


def test_openweather_air_quality_parsing():
    payload = {
        "list": [
            {
                "dt": 1735689600,
                "components": {
                    "pm2_5": 10,
                    "pm10": 20,
                    "co": 200,
                    "no": 1,
                    "no2": 15,
                    "o3": 30,
                    "so2": 5,
                    "nh3": 2,
                },
            }
        ]
    }
    provider = OpenWeatherAirQualityProvider(api_key="fake", client=FakeClient(payload))
    df = provider.fetch_current(LOC)
    assert df["pm2_5"].iloc[0] == 10
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_openweather_empty_list_raises():
    provider = OpenWeatherAirQualityProvider(api_key="fake", client=FakeClient({"list": []}))
    with pytest.raises(ProviderResponseError):
        provider.fetch_current(LOC)


def test_base_http_client_retries_on_rate_limit(requests_mock):
    url = "https://example.test/data"
    requests_mock.get(url, [{"status_code": 429}, {"json": {"ok": True}, "status_code": 200}])
    client = BaseHTTPClient(timeout=5, max_retries=3, backoff_seconds=0.01)
    result = client.get_json(url, {})
    assert result == {"ok": True}
    assert requests_mock.call_count == 2


def test_base_http_client_4xx_raises(requests_mock):
    url = "https://example.test/missing"
    requests_mock.get(url, status_code=404, text="not found")
    client = BaseHTTPClient(timeout=5, max_retries=2, backoff_seconds=0.01)
    with pytest.raises(ProviderResponseError):
        client.get_json(url, {})


def test_hourly_timestamps_are_datetime(sample_observations):
    assert pd.api.types.is_datetime64_any_dtype(sample_observations["timestamp"])
