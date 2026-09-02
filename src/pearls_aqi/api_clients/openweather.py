"""OpenWeather providers (alternative — requires OPENWEATHER_API_KEY).

Implements the Air Pollution API and the Weather/One Call APIs behind the same
interfaces as Open-Meteo. Note the free tier's historical pollution window is
limited; see docs/limitations.md.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from pearls_aqi.api_clients.base import BaseHTTPClient
from pearls_aqi.data.domain import Location
from pearls_aqi.exceptions import ConfigError, ProviderResponseError
from pearls_aqi.utils.logging import get_logger
from pearls_aqi.utils.timeutils import ensure_utc, now_utc

logger = get_logger(__name__)

_AIR_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
_AIR_POLLUTION_HISTORY_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"
_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
_CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# OpenWeather air-pollution components (µg/m³) -> canonical column.
_OW_COMPONENTS = {
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "co": "co",
    "no": "no",
    "no2": "no2",
    "o3": "o3",
    "so2": "so2",
    "nh3": "nh3",
}


def _require_key(api_key: str | None) -> str:
    if not api_key:
        raise ConfigError(
            "OpenWeather selected but OPENWEATHER_API_KEY is not set. "
            "Set it in .env or switch AQI_PROVIDER/WEATHER_PROVIDER to 'openmeteo'."
        )
    return api_key


class OpenWeatherAirQualityProvider:
    name = "openweather"

    def __init__(self, api_key: str | None, client: BaseHTTPClient | None = None) -> None:
        self.api_key = api_key
        self._client = client or BaseHTTPClient()

    def _parse_list(self, payload: dict) -> pd.DataFrame:
        items = payload.get("list") or []
        if not items:
            raise ProviderResponseError("OpenWeather air_pollution returned an empty list.")
        rows = []
        for item in items:
            comp = item.get("components", {})
            row = {"timestamp": pd.to_datetime(item["dt"], unit="s", utc=True)}
            for ow_key, canonical in _OW_COMPONENTS.items():
                row[canonical] = comp.get(ow_key)
            rows.append(row)
        df = pd.DataFrame(rows)
        df["data_source"] = self.name
        return df

    def fetch_current(self, location: Location) -> pd.DataFrame:
        key = _require_key(self.api_key)
        params = {"lat": location.latitude, "lon": location.longitude, "appid": key}
        return self._parse_list(self._client.get_json(_AIR_POLLUTION_URL, params)).tail(1)

    def fetch_historical(self, location: Location, start: datetime, end: datetime) -> pd.DataFrame:
        key = _require_key(self.api_key)
        params = {
            "lat": location.latitude,
            "lon": location.longitude,
            "start": int(ensure_utc(start).timestamp()),
            "end": int(ensure_utc(end).timestamp()),
            "appid": key,
        }
        return self._parse_list(self._client.get_json(_AIR_POLLUTION_HISTORY_URL, params))


class OpenWeatherWeatherProvider:
    name = "openweather"

    def __init__(self, api_key: str | None, client: BaseHTTPClient | None = None) -> None:
        self.api_key = api_key
        self._client = client or BaseHTTPClient()

    def _parse_forecast(self, payload: dict) -> pd.DataFrame:
        items = payload.get("list") or []
        if not items:
            raise ProviderResponseError("OpenWeather forecast returned an empty list.")
        rows = []
        for item in items:
            main = item.get("main", {})
            wind = item.get("wind", {})
            weather = (item.get("weather") or [{}])[0]
            rows.append(
                {
                    "timestamp": pd.to_datetime(item["dt"], unit="s", utc=True),
                    "temperature": main.get("temp"),
                    "feels_like": main.get("feels_like"),
                    "humidity": main.get("humidity"),
                    "pressure": main.get("pressure"),
                    "wind_speed": wind.get("speed"),
                    "wind_direction": wind.get("deg"),
                    "wind_gust": wind.get("gust"),
                    "precipitation": (item.get("rain") or {}).get("3h", 0.0),
                    "rain": (item.get("rain") or {}).get("3h", 0.0),
                    "cloud_cover": (item.get("clouds") or {}).get("all"),
                    "visibility": item.get("visibility"),
                    "dew_point": None,
                    "weather_condition": weather.get("main"),
                    "boundary_layer_height": None,
                }
            )
        df = pd.DataFrame(rows)
        df["data_source"] = self.name
        # OpenWeather 5-day forecast is 3-hourly; upsample to hourly by interpolation.
        return _to_hourly(df)

    def fetch_current(self, location: Location) -> pd.DataFrame:
        key = _require_key(self.api_key)
        params = {
            "lat": location.latitude,
            "lon": location.longitude,
            "units": "metric",
            "appid": key,
        }
        payload = self._client.get_json(_CURRENT_WEATHER_URL, params)
        main = payload.get("main", {})
        wind = payload.get("wind", {})
        weather = (payload.get("weather") or [{}])[0]
        row = {
            "timestamp": pd.to_datetime(payload["dt"], unit="s", utc=True),
            "temperature": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "pressure": main.get("pressure"),
            "wind_speed": wind.get("speed"),
            "wind_direction": wind.get("deg"),
            "wind_gust": wind.get("gust"),
            "precipitation": (payload.get("rain") or {}).get("1h", 0.0),
            "rain": (payload.get("rain") or {}).get("1h", 0.0),
            "cloud_cover": (payload.get("clouds") or {}).get("all"),
            "visibility": payload.get("visibility"),
            "dew_point": None,
            "weather_condition": weather.get("main"),
            "boundary_layer_height": None,
            "data_source": self.name,
        }
        return pd.DataFrame([row])

    def fetch_historical(self, location: Location, start: datetime, end: datetime) -> pd.DataFrame:
        # The free OpenWeather tier does not expose bulk historical hourly weather.
        raise ProviderResponseError(
            "OpenWeather free tier does not provide bulk historical hourly weather. "
            "Use WEATHER_PROVIDER=openmeteo for historical backfill, or a supplied CSV."
        )

    def fetch_forecast(self, location: Location, horizon_hours: int) -> pd.DataFrame:
        key = _require_key(self.api_key)
        params = {
            "lat": location.latitude,
            "lon": location.longitude,
            "units": "metric",
            "appid": key,
        }
        df = self._parse_forecast(self._client.get_json(_FORECAST_URL, params))
        now = ensure_utc(now_utc()).replace(minute=0, second=0, microsecond=0)
        return df[df["timestamp"] > now].head(horizon_hours).reset_index(drop=True)


def _to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a 3-hourly frame to hourly via time interpolation."""
    if df.empty:
        return df
    df = df.sort_values("timestamp").set_index("timestamp")
    numeric = df.select_dtypes("number")
    hourly = numeric.resample("1h").interpolate(method="time")
    # forward-fill non-numeric (weather_condition, data_source)
    non_numeric = df.drop(columns=numeric.columns).resample("1h").ffill()
    out = hourly.join(non_numeric).reset_index()
    return out
