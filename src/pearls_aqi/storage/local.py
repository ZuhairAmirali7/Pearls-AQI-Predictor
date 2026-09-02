"""Local Parquet feature store (default backend).

One Parquet file per feature group under ``local_dir``. Writes are idempotent
upserts keyed on ``(city_id, timestamp)``. Good enough for local development,
CI, and single-node deployments; swap to Hopsworks for multi-consumer/cloud use.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from pearls_aqi.storage.base import FEATURES_GROUP, upsert
from pearls_aqi.utils.io import ensure_dir, read_parquet, write_parquet
from pearls_aqi.utils.logging import get_logger
from pearls_aqi.utils.timeutils import ensure_utc

logger = get_logger(__name__)


class LocalParquetFeatureStore:
    """Filesystem-backed feature store."""

    def __init__(self, local_dir: str | Path = "data/local/feature_store") -> None:
        self.local_dir = ensure_dir(local_dir)

    def _path(self, group: str) -> Path:
        return self.local_dir / f"{group}.parquet"

    def write_features(self, dataframe: pd.DataFrame, group: str = FEATURES_GROUP) -> int:
        if dataframe is None or dataframe.empty:
            logger.warning("write_features called with empty frame for group=%s", group)
            return 0
        path = self._path(group)
        existing = read_parquet(path)
        merged = upsert(existing, dataframe)
        write_parquet(merged, path)
        written = len(dataframe)
        logger.info(
            "Wrote %d rows to feature group '%s' (total now %d) at %s",
            written,
            group,
            len(merged),
            path,
        )
        return written

    def read_features(
        self,
        group: str = FEATURES_GROUP,
        start: datetime | None = None,
        end: datetime | None = None,
        city_id: str | None = None,
    ) -> pd.DataFrame:
        df = read_parquet(self._path(group))
        if df.empty:
            return df
        if city_id is not None and "city_id" in df.columns:
            df = df[df["city_id"] == city_id]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            if start is not None:
                df = df[df["timestamp"] >= ensure_utc(start)]
            if end is not None:
                df = df[df["timestamp"] <= ensure_utc(end)]
            df = df.sort_values(
                ["city_id", "timestamp"] if "city_id" in df.columns else "timestamp"
            )
        return df.reset_index(drop=True)

    def get_training_data(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        city_id: str | None = None,
    ) -> pd.DataFrame:
        return self.read_features(FEATURES_GROUP, start=start, end=end, city_id=city_id)

    def get_latest_features(
        self, city_id: str, n: int = 1, group: str = FEATURES_GROUP
    ) -> pd.DataFrame:
        df = self.read_features(group, city_id=city_id)
        if df.empty:
            return df
        return df.tail(n).reset_index(drop=True)
