"""Model-registry factory with graceful fallback to the local backend."""

from __future__ import annotations

from pearls_aqi.config.models import AppConfig
from pearls_aqi.registry.base import ModelRegistry
from pearls_aqi.registry.local import LocalModelRegistry
from pearls_aqi.utils.logging import get_logger

logger = get_logger(__name__)


def get_model_registry(config: AppConfig, backend: str | None = None) -> ModelRegistry:
    chosen = (backend or config.model_registry.backend).lower()
    if chosen == "hopsworks":
        try:
            from pearls_aqi.registry.hopsworks_registry import HopsworksModelRegistry

            return HopsworksModelRegistry(config)
        except Exception as exc:
            logger.warning("Hopsworks model registry unavailable (%s); falling back to local.", exc)
    return LocalModelRegistry(config.model_registry.local_dir)
