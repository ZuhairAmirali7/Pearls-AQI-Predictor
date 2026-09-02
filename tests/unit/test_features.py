"""Unit tests for feature engineering (incl. a data-leakage guard)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pearls_aqi.config.models import FeaturesConfig
from pearls_aqi.features.engineering import (
    add_change_features,
    add_cyclical_features,
    add_lag_features,
    add_time_features,
    build_features,
    select_feature_columns,
)
from pearls_aqi.features.targets import build_direct_targets, target_columns


@pytest.fixture
def small_frame():
    idx = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "city_id": "test_city",
            "aqi": np.arange(48, dtype=float),
            "pm2_5": np.arange(48, dtype=float) + 10,
            "wind_direction": np.linspace(0, 359, 48),
            "temperature": np.linspace(20, 30, 48),
            "humidity": np.linspace(40, 80, 48),
        }
    )


def test_time_features(small_frame):
    out = add_time_features(small_frame)
    for col in [
        "hour",
        "day_of_week",
        "month",
        "day_of_year",
        "week_of_year",
        "is_weekend",
        "season",
    ]:
        assert col in out.columns
    assert out["hour"].between(0, 23).all()


def test_cyclical_encoding_unit_circle(small_frame):
    out = add_cyclical_features(add_time_features(small_frame))
    assert np.allclose(out["sin_hour"] ** 2 + out["cos_hour"] ** 2, 1.0)
    assert np.allclose(out["sin_month"] ** 2 + out["cos_month"] ** 2, 1.0)
    assert {"wind_dir_sin", "wind_dir_cos"} <= set(out.columns)


def test_lag_features_shift_backward(small_frame):
    out = add_lag_features(small_frame, ["aqi"], [1, 3])
    # aqi is 0..47; lag_1h at row i equals aqi at row i-1.
    assert out["aqi_lag_1h"].iloc[5] == small_frame["aqi"].iloc[4]
    assert out["aqi_lag_3h"].iloc[10] == small_frame["aqi"].iloc[7]
    assert pd.isna(out["aqi_lag_1h"].iloc[0])


def test_change_and_acceleration(small_frame):
    out = add_change_features(small_frame)
    assert "aqi_change_1h" in out.columns
    assert "aqi_acceleration" in out.columns
    # aqi increases by 1 each hour -> change is constant 1, acceleration ~0.
    assert np.allclose(out["aqi_change_1h"].dropna(), 1.0)
    assert np.allclose(out["aqi_acceleration"].dropna(), 0.0)


def test_build_features_produces_families(sample_observations):
    cfg = FeaturesConfig(
        lag_hours=[1, 3, 24],
        rolling_windows=[3, 24],
        rolling_stats=["mean", "std"],
        lag_pollutants=["aqi", "pm2_5"],
    )
    out = build_features(sample_observations, cfg)
    assert any(c.startswith("aqi_lag_") for c in out.columns)
    assert any("roll_mean" in c for c in out.columns)
    assert "sin_hour" in out.columns
    assert "temp_x_humidity" in out.columns


def test_no_future_leakage_in_feature_columns(sample_observations):
    cfg = FeaturesConfig(lag_hours=[1, 3], rolling_windows=[3], rolling_stats=["mean"])
    feats = build_features(sample_observations, cfg)
    with_targets = build_direct_targets(feats, horizon_hours=6)
    feature_cols = select_feature_columns(with_targets, target_columns(6))
    # No target column, and no metadata/identifier, may appear as a feature.
    assert not any(c.startswith("target_") for c in feature_cols)
    for banned in ["city_id", "timestamp", "data_source", "ingested_at", "dominant_pollutant"]:
        assert banned not in feature_cols
