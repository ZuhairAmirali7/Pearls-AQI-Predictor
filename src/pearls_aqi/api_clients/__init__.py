"""External data providers (weather + air quality) behind common interfaces."""

from __future__ import annotations

from pearls_aqi.api_clients.base import AirQualityProvider, WeatherProvider
from pearls_aqi.api_clients.factory import (
    get_air_quality_provider,
    get_weather_provider,
)

__all__ = [
    "AirQualityProvider",
    "WeatherProvider",
    "get_air_quality_provider",
    "get_weather_provider",
]
