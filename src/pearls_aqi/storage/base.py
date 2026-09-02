"""Feature-store interface + shared upsert logic.

The storage layer is intentionally narrow so the backend (local Parquet vs
Hopsworks) can be swapped via ``FEATURE_STORE_BACKEND`` without touching callers.

Feature groups (names/versions come from config):
  raw_air_quality, raw_weather, aqi_features, forecast_inputs,
  model_predictions, model_monitoring
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd

from pearls_aqi.data.schemas import PRIMARY_KEY

# Logical group keys used by callers (mapped to physical names/versions in config).
FEATURES_GROUP = "aqi_features"
RAW_AIR_QUALITY_GROUP = "raw_air_quality"
RAW_WEATHER_GROUP = "raw_weather"
FORECAST_INPUTS_GROUP = "forecast_inputs"
PREDICTIONS_GROUP = "predictions"
MONITORING_GROUP = "monitoring"


@runtime_checkable
class FeatureStoreRepository(Protocol):
    """Read/write processed features and related groups."""

    def write_features(self, dataframe: pd.DataFrame, group: str = FEATURES_GROUP) -> int:
        """Upsert rows into ``group`` by primary key; return rows written."""
        ...

    def read_features(
        self,
        group: str = FEATURES_GROUP,
        start: datetime | None = None,
        end: datetime | None = None,
        city_id: str | None = None,
    ) -> pd.DataFrame: ...

    def get_training_data(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        city_id: str | None = None,
    ) -> pd.DataFrame: ...

    def get_latest_features(
        self, city_id: str, n: int = 1, group: str = FEATURES_GROUP
    ) -> pd.DataFrame: ...


def upsert(existing: pd.DataFrame, new: pd.DataFrame, key: list[str] | None = None) -> pd.DataFrame:
    """Idempotent upsert: concat, keep the last write per primary key, sort.

    Deterministic primary keys (``city_id`` + ``timestamp``) guarantee re-running
    a pipeline over the same period never creates duplicates.
    """
    key = key or PRIMARY_KEY
    if existing is None or existing.empty:
        combined = new.copy()
    elif new is None or new.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, new], ignore_index=True)
    present_keys = [k for k in key if k in combined.columns]
    if present_keys:
        combined = combined.drop_duplicates(subset=present_keys, keep="last")
        if "timestamp" in combined.columns:
            combined = combined.sort_values(present_keys).reset_index(drop=True)
    return combined
