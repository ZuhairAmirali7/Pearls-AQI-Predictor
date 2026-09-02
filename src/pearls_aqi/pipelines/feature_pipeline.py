"""Feature pipeline: fetch → validate → merge → AQI → features → store.

Idempotent: rows are keyed on ``(city_id, timestamp)`` and upserted, so running
the same window twice never creates duplicates. Logs fetched / transformed /
rejected / written counts every run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from pearls_aqi.config.models import AppConfig
from pearls_aqi.data.domain import Location
from pearls_aqi.data.schemas import METADATA_COLUMNS
from pearls_aqi.data.validation import ValidationReport, validate_observations
from pearls_aqi.features.engineering import build_features, compute_aqi_columns, merge_observations
from pearls_aqi.storage.base import FEATURES_GROUP, RAW_AIR_QUALITY_GROUP, RAW_WEATHER_GROUP
from pearls_aqi.utils.logging import get_logger
from pearls_aqi.utils.timeutils import ensure_utc, now_utc, utc_floor_hour

logger = get_logger(__name__)


@dataclass
class FeatureRunSummary:
    city: str
    start: datetime | None
    end: datetime | None
    fetched: int = 0
    transformed: int = 0
    rejected: int = 0
    written: int = 0
    validation: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "city": self.city,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "fetched": self.fetched,
            "transformed": self.transformed,
            "rejected": self.rejected,
            "written": self.written,
            "validation": self.validation,
        }


def attach_metadata(
    df: pd.DataFrame, location: Location, source: str, version: str
) -> pd.DataFrame:
    """Attach provenance columns required on every stored record."""
    df = df.copy()
    df["city_id"] = location.city_id
    df["city"] = location.city
    df["country"] = location.country
    df["latitude"] = location.latitude
    df["longitude"] = location.longitude
    df["timezone"] = location.timezone
    df["data_source"] = source
    df["ingested_at"] = ensure_utc(now_utc())
    df["feature_pipeline_version"] = version
    ordered = METADATA_COLUMNS + [c for c in df.columns if c not in METADATA_COLUMNS]
    return df[ordered]


class FeaturePipeline:
    def __init__(
        self,
        config: AppConfig,
        aq_provider: Any = None,
        weather_provider: Any = None,
        feature_store: Any = None,
    ) -> None:
        self.config = config
        self._aq = aq_provider
        self._wx = weather_provider
        self._fs = feature_store

    @property
    def aq_provider(self):
        if self._aq is None:
            from pearls_aqi.api_clients import get_air_quality_provider

            self._aq = get_air_quality_provider(self.config)
        return self._aq

    @property
    def weather_provider(self):
        if self._wx is None:
            from pearls_aqi.api_clients import get_weather_provider

            self._wx = get_weather_provider(self.config)
        return self._wx

    @property
    def feature_store(self):
        if self._fs is None:
            from pearls_aqi.storage import get_feature_store

            self._fs = get_feature_store(self.config)
        return self._fs

    # -- core -------------------------------------------------------------
    def ingest_window(
        self, location: Location, start: datetime, end: datetime
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Fetch + merge + compute AQI for a window; return merged frame + report."""
        aq = self.aq_provider.fetch_historical(location, start, end)
        wx = self.weather_provider.fetch_historical(location, start, end)
        merged = merge_observations(aq, wx)
        if merged.empty:
            return merged, validate_observations(merged, allow_empty=True)
        if self.config.data.target_source == "local_computed":
            merged = compute_aqi_columns(merged, standard=self.config.data.aqi_standard)
        merged = attach_metadata(
            merged, location, self.aq_provider.name, self.config.project.feature_pipeline_version
        )
        report = validate_observations(merged, required_columns=["city_id", "timestamp", "aqi"])
        return merged, report

    def run(
        self,
        location: Location,
        start: datetime,
        end: datetime,
        write: bool = True,
        store_raw: bool | None = None,
    ) -> FeatureRunSummary:
        """Process a window and write engineered features to the store."""
        start, end = ensure_utc(start), ensure_utc(end)
        summary = FeatureRunSummary(city=location.city, start=start, end=end)
        merged, report = self.ingest_window(location, start, end)
        summary.fetched = len(merged)
        summary.validation = report.as_dict()

        if merged.empty:
            logger.warning("Feature pipeline: no data fetched for %s in window.", location.city)
            return summary
        if not report.ok:
            summary.rejected = summary.fetched
            report.raise_if_failed()

        features = build_features(
            merged,
            self.config.features,
            location=location,
            recompute_aqi=False,
            aqi_standard=self.config.data.aqi_standard,
        )
        summary.transformed = len(features)

        store_raw = self.config.providers.store_raw_payloads if store_raw is None else store_raw
        if write:
            if store_raw:
                self.feature_store.write_features(merged, group=RAW_AIR_QUALITY_GROUP)
                self.feature_store.write_features(merged, group=RAW_WEATHER_GROUP)
            summary.written = self.feature_store.write_features(features, group=FEATURES_GROUP)
        logger.info(
            "Feature pipeline for %s: fetched=%d transformed=%d rejected=%d written=%d",
            location.city,
            summary.fetched,
            summary.transformed,
            summary.rejected,
            summary.written,
        )
        return summary

    def run_current(
        self, location: Location, lookback_hours: int | None = None
    ) -> FeatureRunSummary:
        """Hourly job: process a recent window big enough for lag/rolling features."""
        max_lag = max(self.config.features.lag_hours + self.config.features.rolling_windows)
        lookback = lookback_hours or (max_lag + 48)
        end = utc_floor_hour(now_utc())
        start = end - timedelta(hours=lookback)
        return self.run(location, start, end)
