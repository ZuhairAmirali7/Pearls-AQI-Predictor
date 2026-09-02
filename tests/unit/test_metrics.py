"""Unit tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from pearls_aqi.evaluation.metrics import (
    bias,
    hazard_metrics,
    mae,
    metrics_by_horizon,
    r2,
    regression_metrics,
    rmse,
)


def test_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert r2(y, y) == 1.0


def test_known_mae_rmse_bias():
    yt = np.array([10.0, 20.0, 30.0])
    yp = np.array([12.0, 18.0, 33.0])  # errors +2, -2, +3
    assert mae(yt, yp) == (2 + 2 + 3) / 3
    assert abs(rmse(yt, yp) - np.sqrt((4 + 4 + 9) / 3)) < 1e-9
    assert abs(bias(yt, yp) - (2 - 2 + 3) / 3) < 1e-9


def test_nan_pairs_ignored():
    yt = np.array([10.0, np.nan, 30.0])
    yp = np.array([12.0, 5.0, 30.0])
    assert mae(yt, yp) == 1.0  # only two valid pairs: |2| and |0|


def test_regression_metrics_keys():
    m = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert {"rmse", "mae", "r2", "smape", "mape", "median_ae", "bias", "n"} <= set(m)


def test_metrics_by_horizon_shape():
    yt = np.random.default_rng(0).normal(100, 20, size=(50, 6))
    yp = yt + np.random.default_rng(1).normal(0, 5, size=(50, 6))
    df = metrics_by_horizon(yt, yp)
    assert len(df) == 6
    assert list(df["horizon_hour"]) == [1, 2, 3, 4, 5, 6]


def test_hazard_metrics_threshold():
    yt = np.array([100.0, 200.0, 300.0])
    yp = np.array([110.0, 210.0, 280.0])
    hm = hazard_metrics(yt, yp, threshold=151)
    assert hm["hazard_n"] == 2  # 200 and 300 exceed the threshold
    assert hm["hazard_mae"] == (10 + 20) / 2
