"""Shared FastAPI dependencies (cached config + services).

Services are built once per process and reused across requests. Tests can
override these via ``app.dependency_overrides``.
"""

from __future__ import annotations

from functools import lru_cache

from pearls_aqi.config import AppConfig, load_config
from pearls_aqi.data.domain import Location
from pearls_aqi.forecasting import ForecastService
from pearls_aqi.monitoring import MonitoringService
from pearls_aqi.utils.env import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return load_config()


@lru_cache(maxsize=1)
def get_feature_store():
    from pearls_aqi.storage import get_feature_store as _build

    return _build(get_config())


@lru_cache(maxsize=1)
def get_registry():
    from pearls_aqi.registry import get_model_registry

    return get_model_registry(get_config())


@lru_cache(maxsize=1)
def get_forecast_service() -> ForecastService:
    return ForecastService(get_config(), feature_store=get_feature_store(), registry=get_registry())


@lru_cache(maxsize=1)
def get_monitoring_service() -> MonitoringService:
    return MonitoringService(get_config(), feature_store=get_feature_store())


def resolve_location(city: str | None) -> Location:
    """Resolve a city name to a Location (active location if None)."""
    config = get_config()
    if city is None or city == config.location.city:
        return config.active_location
    return config.location_for_city(city)
