"""Model registry abstraction: local filesystem (default) or Hopsworks."""

from __future__ import annotations

from pearls_aqi.registry.base import (
    ModelMetadata,
    ModelRegistry,
    ModelStatus,
)
from pearls_aqi.registry.factory import get_model_registry

__all__ = ["ModelMetadata", "ModelRegistry", "ModelStatus", "get_model_registry"]
