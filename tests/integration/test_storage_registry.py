"""Integration tests: local feature store + model registry lifecycle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pearls_aqi.models import build_model
from pearls_aqi.registry.base import ModelMetadata, ModelStatus
from pearls_aqi.storage.base import FEATURES_GROUP

pytestmark = pytest.mark.integration


def test_feature_store_round_trip(tmp_feature_store):
    df = pd.DataFrame(
        {
            "city_id": ["c"] * 3,
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"),
            "aqi": [50.0, 60.0, 70.0],
        }
    )
    written = tmp_feature_store.write_features(df, group=FEATURES_GROUP)
    assert written == 3
    back = tmp_feature_store.read_features(FEATURES_GROUP, city_id="c")
    assert len(back) == 3
    latest = tmp_feature_store.get_latest_features("c", n=1)
    assert latest["aqi"].iloc[0] == 70.0


def _fitted_model(horizon=6):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"aqi": rng.uniform(20, 100, 30), "hour": rng.integers(0, 24, 30)})
    y = pd.DataFrame(rng.uniform(20, 100, size=(30, horizon)))
    model = build_model("persistence", horizon)
    model.fit(X, y)
    return model


def _metadata(version=0):
    return ModelMetadata(
        model_name="aqi_forecast_test",
        version=version,
        model_type="persistence",
        city_id="test",
        metrics={"mae": 10.0, "validation_mae": 10.0},
        features=["aqi", "hour"],
    )


def test_registry_promotes_and_loads_approved_not_newest(tmp_registry):
    # v1 approved.
    v1 = tmp_registry.register(_fitted_model(), _metadata(), status=ModelStatus.CANDIDATE)
    tmp_registry.set_status("aqi_forecast_test", v1.version, ModelStatus.APPROVED)

    # v2 registered later but only as candidate.
    tmp_registry.register(_fitted_model(), _metadata(), status=ModelStatus.CANDIDATE)

    model, meta = tmp_registry.load_latest_approved(model_name="aqi_forecast_test", city_id="test")
    assert meta.status == ModelStatus.APPROVED.value
    assert meta.version == v1.version  # approved v1, NOT the newer candidate v2
    assert model is not None


def test_registry_lists_versions(tmp_registry):
    tmp_registry.register(_fitted_model(), _metadata(), status=ModelStatus.CANDIDATE)
    versions = tmp_registry.list_versions("aqi_forecast_test")
    assert len(versions) >= 1
    assert versions[0].model_type == "persistence"
