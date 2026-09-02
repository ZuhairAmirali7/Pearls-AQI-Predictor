"""Unit tests for forecast-target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pearls_aqi.features.targets import (
    build_direct_targets,
    target_columns,
    trainable_rows,
)


def _frame(n=100):
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": idx, "city_id": "c", "aqi": np.arange(n, dtype=float)})


def test_target_columns_length():
    assert len(target_columns(72)) == 72
    assert target_columns(3) == [
        "target_aqi_t_plus_1",
        "target_aqi_t_plus_2",
        "target_aqi_t_plus_3",
    ]


def test_targets_shift_backward():
    df = build_direct_targets(_frame(50), horizon_hours=3)
    # Row i target_t_plus_1 == aqi at row i+1.
    assert df["target_aqi_t_plus_1"].iloc[0] == 1.0
    assert df["target_aqi_t_plus_3"].iloc[0] == 3.0
    # Last rows have NaN targets (no future data).
    assert pd.isna(df["target_aqi_t_plus_1"].iloc[-1])


def test_trainable_rows_drops_missing_targets():
    df = build_direct_targets(_frame(50), horizon_hours=6)
    trainable = trainable_rows(df, horizon_hours=6)
    # Last 6 rows lack complete targets and are dropped.
    assert len(trainable) == 44
    assert not trainable[target_columns(6)].isna().any().any()
