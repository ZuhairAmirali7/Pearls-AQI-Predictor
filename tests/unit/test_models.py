"""Unit tests for models and the model factory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pearls_aqi.config.models import TrainingConfig
from pearls_aqi.models import build_model, is_available
from pearls_aqi.models.baselines import BASELINE_MODELS

HORIZON = 6


def _xy(n=40):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "aqi": rng.uniform(20, 150, n),
            "hour": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "pm2_5": rng.uniform(10, 120, n),
            "aqi_lag_24h": rng.uniform(20, 150, n),
            "aqi_roll_mean_24h": rng.uniform(20, 150, n),
        }
    )
    y = pd.DataFrame(
        rng.uniform(20, 150, size=(n, HORIZON)),
        columns=[f"target_aqi_t_plus_{h}" for h in range(1, HORIZON + 1)],
    )
    return X, y


@pytest.mark.parametrize("name", list(BASELINE_MODELS))
def test_baseline_predict_shape(name):
    X, y = _xy()
    model = build_model(name, HORIZON)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X), HORIZON)
    assert np.all((preds >= 0) & (preds <= 500))


@pytest.mark.parametrize(
    "name", ["ridge", "elastic_net", "random_forest", "hist_gradient_boosting"]
)
def test_sklearn_models_fit_predict(name):
    X, y = _xy()
    model = build_model(name, HORIZON, TrainingConfig(), tune=False)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X), HORIZON)
    assert np.all(np.isfinite(preds))


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("does_not_exist", HORIZON)


def test_is_available_returns_bool():
    assert isinstance(is_available("tensorflow"), bool)
    assert is_available("ridge") is True


def test_model_persistence_roundtrip(tmp_path):
    X, y = _xy()
    model = build_model("ridge", HORIZON, TrainingConfig(), tune=False)
    model.fit(X, y)
    model.save(tmp_path)
    from pearls_aqi.models import load_model

    restored = load_model("ridge", tmp_path)
    assert np.allclose(restored.predict(X), model.predict(X))
