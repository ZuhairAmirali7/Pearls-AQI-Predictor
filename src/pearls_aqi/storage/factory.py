"""Feature-store factory with graceful fallback to local Parquet."""

from __future__ import annotations

from pearls_aqi.config.models import AppConfig
from pearls_aqi.storage.base import FeatureStoreRepository
from pearls_aqi.storage.local import LocalParquetFeatureStore
from pearls_aqi.utils.logging import get_logger

logger = get_logger(__name__)


def get_feature_store(config: AppConfig, backend: str | None = None) -> FeatureStoreRepository:
    """Return the configured feature store.

    If ``hopsworks`` is requested but unavailable, log a warning and fall back to
    the local Parquet store so pipelines keep running (graceful degradation).
    """
    chosen = (backend or config.feature_store.backend).lower()
    if chosen == "hopsworks":
        try:
            from pearls_aqi.storage.hopsworks_store import HopsworksFeatureStore

            return HopsworksFeatureStore(config)
        except Exception as exc:
            logger.warning(
                "Hopsworks feature store unavailable (%s); falling back to local Parquet.", exc
            )
    return LocalParquetFeatureStore(config.feature_store.local_dir)
