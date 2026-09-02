"""Provider factories — build the configured provider from settings + env."""

from __future__ import annotations

import os

from pearls_aqi.api_clients.base import (
    AirQualityProvider,
    BaseHTTPClient,
    WeatherProvider,
)
from pearls_aqi.api_clients.open_meteo import (
    OpenMeteoAirQualityProvider,
    OpenMeteoWeatherProvider,
)
from pearls_aqi.api_clients.openweather import (
    OpenWeatherAirQualityProvider,
    OpenWeatherWeatherProvider,
)
from pearls_aqi.api_clients.sample_provider import (
    SampleAirQualityProvider,
    SampleWeatherProvider,
)
from pearls_aqi.config.models import AppConfig
from pearls_aqi.exceptions import ConfigError


def _http_client(config: AppConfig) -> BaseHTTPClient:
    p = config.providers
    return BaseHTTPClient(
        timeout=p.request_timeout_seconds,
        max_retries=p.max_retries,
        backoff_seconds=p.backoff_seconds,
    )


def get_air_quality_provider(config: AppConfig, name: str | None = None) -> AirQualityProvider:
    provider = (name or config.providers.air_quality).lower()
    if provider == "openmeteo":
        return OpenMeteoAirQualityProvider(client=_http_client(config))
    if provider == "openweather":
        return OpenWeatherAirQualityProvider(
            api_key=os.getenv("OPENWEATHER_API_KEY"), client=_http_client(config)
        )
    if provider == "sample":
        return SampleAirQualityProvider()
    raise ConfigError(f"Unknown air-quality provider: {provider!r}")


def get_weather_provider(config: AppConfig, name: str | None = None) -> WeatherProvider:
    provider = (name or config.providers.weather).lower()
    if provider == "openmeteo":
        return OpenMeteoWeatherProvider(client=_http_client(config))
    if provider == "openweather":
        return OpenWeatherWeatherProvider(
            api_key=os.getenv("OPENWEATHER_API_KEY"), client=_http_client(config)
        )
    if provider == "sample":
        return SampleWeatherProvider()
    raise ConfigError(f"Unknown weather provider: {provider!r}")
