"""Edge-case coverage for evaluation metrics."""

from __future__ import annotations

import numpy as np

from pearls_aqi.evaluation.metrics import (
    category_confusion,
    mae,
    mape,
    median_abs_error,
    metrics_by_category,
    r2,
    regression_metrics,
    rmse,
    smape,
)


def test_empty_arrays_return_nan():
    empty = np.array([])
    for fn in (mae, rmse, r2, smape, mape, median_abs_error):
        assert np.isnan(fn(empty, empty))


def test_smape_and_mape_known_values():
    yt = np.array([100.0, 100.0])
    assert smape(yt, yt) == 0.0
    assert mape(yt, yt) == 0.0
    # 10% error on both points.
    assert abs(mape(yt, np.array([110.0, 90.0])) - 10.0) < 1e-9


def test_r2_constant_truth_is_nan():
    # Zero variance in truth -> R² undefined (nan).
    assert np.isnan(r2(np.array([5.0, 5.0, 5.0]), np.array([5.0, 6.0, 4.0])))


def test_regression_metrics_all_nan_inputs():
    m = regression_metrics(np.array([np.nan, np.nan]), np.array([1.0, 2.0]))
    assert m["n"] == 0
    assert np.isnan(m["mae"])


def test_metrics_by_category():
    yt = np.array([30.0, 30.0, 180.0])  # Good, Good, Unhealthy
    yp = np.array([35.0, 25.0, 200.0])
    out = metrics_by_category(yt, yp)
    assert set(out.columns) == {"category", "n", "mae"}
    good = out[out["category"] == "Good"]
    assert good["n"].iloc[0] == 2


def test_metrics_by_category_empty():
    out = metrics_by_category(np.array([]), np.array([]))
    assert out.empty


def test_category_confusion_is_square():
    yt = np.array([30.0, 180.0, 320.0])
    yp = np.array([40.0, 160.0, 310.0])
    cm = category_confusion(yt, yp)
    assert cm.shape == (6, 6)  # six EPA categories
    assert cm.to_numpy().sum() == 3
