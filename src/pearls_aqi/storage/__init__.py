"""Feature-store abstraction: local Parquet (default) or Hopsworks."""

from __future__ import annotations

from pearls_aqi.storage.base import FeatureStoreRepository
from pearls_aqi.storage.factory import get_feature_store

__all__ = ["FeatureStoreRepository", "get_feature_store"]
