"""Open-Meteo providers (default — free, no API key required).

- Air quality:   https://air-quality-api.open-meteo.com/v1/air-quality
- Weather (fcst): https://api.open-meteo.com/v1/forecast
- Weather (hist): https://archive-api.open-meteo.com/v1/archive  (ERA5 reanalysis)

All requests use ``timezone=UTC`` and ``wind_speed_unit=ms``. Data licensed
CC BY 4.0 by Open-Meteo.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from pearls_aqi.api_clients.base import BaseHTTPClient, hourly_frame_from_arrays
from pearls_aqi.data.domain import Location
from pearls_aqi.exceptions import ProviderResponseError
from pearls_aqi.utils.logging import get_logger
from pearls_aqi.utils.timeutils import ensure_utc, now_utc

logger = get_logger(__name__)

_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo air-quality variable -> canonical column.
_AQ_FIELDS = {
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "carbon_monoxide": "co",
    "nitrogen_monoxide": "no",
    "nitrogen_dioxide": "no2",
    "ozone": "o3",
    "sulphur_dioxide": "so2",
    "ammonia": "nh3",
}

# Open-Meteo weather variable -> canonical column.
_WX_FIELDS = {
    "temperature_2m": "temperature",
    "apparent_temperature": "feels_like",
    "relative_humidity_2m": "humidity",
    "surface_pressure": "pressure",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "wind_gusts_10m": "wind_gust",
    "precipitation": "precipitation",
    "rain": "rain",
    "cloud_cover": "cloud_cover",
    "visibility": "visibility",
    "dew_point_2m": "dew_point",
    "weather_code": "weather_condition",
}

# Coarse WMO weather-code -> label mapping.
_WMO = {
    0: "Clear",
    1: "Mainly Clear",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Fog",
    51: "Drizzle",
    53: "Drizzle",
    55: "Drizzle",
    61: "Rain",
    63: "Rain",
    65: "Rain",
    71: "Snow",
    80: "Rain Showers",
    81: "Rain Showers",
    82: "Rain Showers",
    95: "Thunderstorm",
    96: "Thunderstorm",
    99: "Thunderstorm",
}


def _extract_hourly(payload: dict, source: str) -> dict:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ProviderResponseError(f"Open-Meteo payload missing 'hourly' block from {source}.")
    return hourly


class OpenMeteoAirQualityProvider:
    """Air-quality provider backed by Open-Meteo."""

    name = "openmeteo"

    def __init__(self, client: BaseHTTPClient | None = None) -> None:
        self._client = client or BaseHTTPClient()

    def _base_params(self, location: Location) -> dict:
        return {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "hourly": ",".join(_AQ_FIELDS.keys()),
            "timezone": "UTC",
        }

    def _to_frame(self, payload: dict) -> pd.DataFrame:
        hourly = _extract_hourly(payload, self.name)
        variables = {
            canonical: hourly.get(src, [None] * len(hourly["time"]))
            for src, canonical in _AQ_FIELDS.items()
        }
        return hourly_frame_from_arrays(hourly["time"], variables, self.name)

    def fetch_current(self, location: Location) -> pd.DataFrame:
        params = {**self._base_params(location), "past_days": 1, "forecast_days": 1}
        df = self._to_frame(self._client.get_json(_AQ_URL, params))
        df = df[df["timestamp"] <= ensure_utc(now_utc())]
        return df.tail(1).reset_index(drop=True)

    def fetch_historical(self, location: Location, start: datetime, end: datetime) -> pd.DataFrame:
        params = {
            **self._base_params(location),
            "start_date": ensure_utc(start).date().isoformat(),
            "end_date": ensure_utc(end).date().isoformat(),
        }
        return self._to_frame(self._client.get_json(_AQ_URL, params))


class OpenMeteoWeatherProvider:
    """Weather provider backed by Open-Meteo (forecast + ERA5 archive)."""

    name = "openmeteo"

    def __init__(self, client: BaseHTTPClient | None = None) -> None:
        self._client = client or BaseHTTPClient()

    def _base_params(self, location: Location) -> dict:
        return {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "hourly": ",".join(_WX_FIELDS.keys()),
            "timezone": "UTC",
            "wind_speed_unit": "ms",
        }

    def _to_frame(self, payload: dict) -> pd.DataFrame:
        hourly = _extract_hourly(payload, self.name)
        n = len(hourly["time"])
        variables = {}
        for src, canonical in _WX_FIELDS.items():
            values = hourly.get(src, [None] * n)
            if canonical == "weather_condition":
                values = [_WMO.get(v, "Unknown") if v is not None else None for v in values]
            variables[canonical] = values
        variables["boundary_layer_height"] = hourly.get("boundary_layer_height", [None] * n)
        return hourly_frame_from_arrays(hourly["time"], variables, self.name)

    def fetch_current(self, location: Location) -> pd.DataFrame:
        params = {**self._base_params(location), "past_days": 1, "forecast_days": 1}
        df = self._to_frame(self._client.get_json(_FORECAST_URL, params))
        df = df[df["timestamp"] <= ensure_utc(now_utc())]
        return df.tail(1).reset_index(drop=True)

    def fetch_historical(self, location: Location, start: datetime, end: datetime) -> pd.DataFrame:
        params = {
            **self._base_params(location),
            "start_date": ensure_utc(start).date().isoformat(),
            "end_date": ensure_utc(end).date().isoformat(),
        }
        return self._to_frame(self._client.get_json(_ARCHIVE_URL, params))

    def fetch_forecast(self, location: Location, horizon_hours: int) -> pd.DataFrame:
        forecast_days = max(1, min(16, (horizon_hours + 23) // 24 + 1))
        params = {**self._base_params(location), "forecast_days": forecast_days, "past_days": 1}
        df = self._to_frame(self._client.get_json(_FORECAST_URL, params))
        now = ensure_utc(now_utc()).replace(minute=0, second=0, microsecond=0)
        future = df[df["timestamp"] > now].head(horizon_hours).reset_index(drop=True)
        if future.empty:
            raise ProviderResponseError("Open-Meteo returned no future forecast rows.")
        return future
