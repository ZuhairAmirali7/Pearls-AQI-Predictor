"""Chronological train/validation/test splitting.

Time-series data is **never** shuffled. Rows are ordered by timestamp and cut
into contiguous train → gap → validation → gap → test blocks. The gap (default
= forecast horizon) prevents target windows from straddling a boundary and
leaking future information across splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> dict[str, int]:
        return {
            "train_rows": len(self.train),
            "validation_rows": len(self.validation),
            "test_rows": len(self.test),
        }


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    gap_hours: int = 72,
    time_column: str = "timestamp",
) -> Split:
    """Split a time-ordered frame into train/val/test with leakage gaps."""
    if df.empty:
        return Split(df.copy(), df.copy(), df.copy())
    total = train_fraction + validation_fraction + test_fraction
    if total > 1.0 + 1e-9:
        raise ValueError(f"Fractions sum to {total} > 1.0")

    ordered = df.sort_values(time_column).reset_index(drop=True)
    n = len(ordered)
    n_train = int(n * train_fraction)
    n_val = int(n * validation_fraction)

    train = ordered.iloc[:n_train]
    val_start = min(n, n_train + gap_hours)
    validation = ordered.iloc[val_start : val_start + n_val]
    test_start = min(n, val_start + n_val + gap_hours)
    test = ordered.iloc[test_start:]

    return Split(
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
