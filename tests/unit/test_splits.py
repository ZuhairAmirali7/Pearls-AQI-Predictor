"""Unit tests for chronological splitting (no shuffling, leakage gaps)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pearls_aqi.evaluation.splits import chronological_split


def _frame(n=1000):
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": idx, "value": np.arange(n)})


def test_split_is_chronological_and_gapped():
    df = _frame(1000)
    split = chronological_split(df, 0.7, 0.15, 0.15, gap_hours=24)
    # Ordered and non-overlapping in time.
    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()
    # Gap enforced between train and validation.
    gap = split.validation["timestamp"].min() - split.train["timestamp"].max()
    assert gap >= pd.Timedelta(hours=24)


def test_split_fraction_sizes_reasonable():
    df = _frame(1000)
    split = chronological_split(df, 0.7, 0.15, 0.15, gap_hours=0)
    assert len(split.train) == 700
    assert len(split.validation) == 150


def test_empty_frame_returns_empty_splits():
    split = chronological_split(pd.DataFrame(columns=["timestamp"]))
    assert split.train.empty and split.validation.empty and split.test.empty
