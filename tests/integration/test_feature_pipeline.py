"""Integration test: feature pipeline with sample providers (offline)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from pearls_aqi.api_clients.sample_provider import (
    SampleAirQualityProvider,
    SampleWeatherProvider,
)
from pearls_aqi.pipelines import FeaturePipeline
from pearls_aqi.storage.base import FEATURES_GROUP
from pearls_aqi.utils.timeutils import now_utc, utc_floor_hour

pytestmark = pytest.mark.integration


def test_feature_pipeline_writes_and_is_idempotent(sample_config, location, tmp_feature_store):
    pipeline = FeaturePipeline(
        sample_config, SampleAirQualityProvider(), SampleWeatherProvider(), tmp_feature_store
    )
    end = utc_floor_hour(now_utc())
    start = end - timedelta(days=20)

    summary1 = pipeline.run(location, start, end)
    assert summary1.written > 0
    assert summary1.validation["ok"]
    n1 = len(tmp_feature_store.read_features(FEATURES_GROUP, city_id=location.city_id))

    # Running the same window again must not create duplicate rows.
    pipeline.run(location, start, end)
    stored = tmp_feature_store.read_features(FEATURES_GROUP, city_id=location.city_id)
    assert len(stored) == n1
    assert not stored.duplicated(subset=["city_id", "timestamp"]).any()


def test_features_include_target_base_and_engineered(sample_config, location, tmp_feature_store):
    pipeline = FeaturePipeline(
        sample_config, SampleAirQualityProvider(), SampleWeatherProvider(), tmp_feature_store
    )
    end = utc_floor_hour(now_utc())
    pipeline.run(location, end - timedelta(days=15), end)
    df = tmp_feature_store.read_features(FEATURES_GROUP, city_id=location.city_id)
    assert "aqi" in df.columns
    assert any(c.startswith("aqi_lag_") for c in df.columns)
    assert "sin_hour" in df.columns
