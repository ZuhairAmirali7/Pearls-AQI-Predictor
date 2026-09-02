"""Configuration loading: YAML + environment-variable overrides."""

from __future__ import annotations

from pearls_aqi.config.models import (
    AlertsConfig,
    AppConfig,
    DataConfig,
    FeaturesConfig,
    FeatureStoreConfig,
    ForecastConfig,
    ModelPromotionConfig,
    ModelRegistryConfig,
    MonitoringConfig,
    ProvidersConfig,
    TrainingConfig,
)
from pearls_aqi.config.settings import get_config, load_config

__all__ = [
    "AlertsConfig",
    "AppConfig",
    "DataConfig",
    "FeatureStoreConfig",
    "FeaturesConfig",
    "ForecastConfig",
    "ModelPromotionConfig",
    "ModelRegistryConfig",
    "MonitoringConfig",
    "ProvidersConfig",
    "TrainingConfig",
    "get_config",
    "load_config",
]
