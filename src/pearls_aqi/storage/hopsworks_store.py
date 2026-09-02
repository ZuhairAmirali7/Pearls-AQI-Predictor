"""Hopsworks feature-store backend.

Complete integration code, but exercised in CI only via mocks — it requires a
real Hopsworks account (``HOPSWORKS_API_KEY`` + ``HOPSWORKS_PROJECT``) to run for
real. The ``hopsworks`` package is imported lazily so the project stays runnable
without it installed. When unavailable, callers should fall back to the local
backend (the factory does this automatically unless the backend is forced).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd

from pearls_aqi.config.models import AppConfig
from pearls_aqi.data.schemas import EVENT_TIME, PRIMARY_KEY
from pearls_aqi.exceptions import FeatureStoreError
from pearls_aqi.storage.base import FEATURES_GROUP
from pearls_aqi.utils.logging import get_logger
from pearls_aqi.utils.timeutils import ensure_utc

logger = get_logger(__name__)


class HopsworksFeatureStore:
    """Feature store backed by Hopsworks."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._project: Any = None
        self._fs: Any = None

    # -- connection -------------------------------------------------------
    def _connect(self) -> None:
        if self._fs is not None:
            return
        try:
            import hopsworks
        except ImportError as exc:  # pragma: no cover
            raise FeatureStoreError(
                "hopsworks package not installed. Run `pip install -e '.[hopsworks]'` "
                "or set FEATURE_STORE_BACKEND=local."
            ) from exc

        api_key = os.getenv("HOPSWORKS_API_KEY")
        if not api_key:
            raise FeatureStoreError("HOPSWORKS_API_KEY not set.")
        try:
            self._project = hopsworks.login(
                api_key_value=api_key, project=os.getenv("HOPSWORKS_PROJECT")
            )
            self._fs = self._project.get_feature_store()
            logger.info("Connected to Hopsworks project '%s'.", self._project.name)
        except Exception as exc:  # pragma: no cover — network
            raise FeatureStoreError(f"Failed to connect to Hopsworks: {exc}") from exc

    def _group_meta(self, group: str) -> tuple[str, int]:
        fg = self.config.feature_store.groups.get(group)
        if fg is None:
            return group, 1
        return fg.name, fg.version

    def _get_or_create_fg(self, group: str) -> Any:
        self._connect()
        name, version = self._group_meta(group)
        return self._fs.get_or_create_feature_group(
            name=name,
            version=version,
            primary_key=PRIMARY_KEY,
            event_time=EVENT_TIME,
            description=f"Pearls AQI Predictor group '{group}'.",
            online_enabled=False,
        )

    # -- API --------------------------------------------------------------
    def write_features(self, dataframe: pd.DataFrame, group: str = FEATURES_GROUP) -> int:
        if dataframe is None or dataframe.empty:
            return 0
        fg = self._get_or_create_fg(group)
        try:
            fg.insert(dataframe, write_options={"wait_for_job": True})
        except Exception as exc:  # pragma: no cover — network
            raise FeatureStoreError(f"Hopsworks insert failed for {group}: {exc}") from exc
        logger.info("Inserted %d rows into Hopsworks group '%s'.", len(dataframe), group)
        return len(dataframe)

    def read_features(
        self,
        group: str = FEATURES_GROUP,
        start: datetime | None = None,
        end: datetime | None = None,
        city_id: str | None = None,
    ) -> pd.DataFrame:
        fg = self._get_or_create_fg(group)
        try:
            df = fg.read()
        except Exception as exc:  # pragma: no cover — network
            raise FeatureStoreError(f"Hopsworks read failed for {group}: {exc}") from exc
        if df.empty:
            return df
        df[EVENT_TIME] = pd.to_datetime(df[EVENT_TIME], utc=True)
        if city_id is not None and "city_id" in df.columns:
            df = df[df["city_id"] == city_id]
        if start is not None:
            df = df[df[EVENT_TIME] >= ensure_utc(start)]
        if end is not None:
            df = df[df[EVENT_TIME] <= ensure_utc(end)]
        return df.sort_values(EVENT_TIME).reset_index(drop=True)

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
        return df.tail(n).reset_index(drop=True) if not df.empty else df
