"""Forecast target construction (direct multi-horizon strategy).

For a horizon of H hours we create H targets, ``target_aqi_t_plus_1 ..
target_aqi_t_plus_H``, each the AQI shifted *backward* by the horizon so that the
row at time ``t`` carries the value observed at ``t+h``. A separate model output
predicts each horizon (see :mod:`pearls_aqi.models`), which avoids the error
accumulation of recursive forecasting and lets each horizon use the appropriately
aligned future weather forecast at prediction time.

Rows whose targets fall past the end of the series get ``NaN`` and are excluded
from training — never imputed.
"""

from __future__ import annotations

import pandas as pd

TARGET_PREFIX = "target_aqi_t_plus_"


def target_columns(horizon_hours: int) -> list[str]:
    return [f"{TARGET_PREFIX}{h}" for h in range(1, horizon_hours + 1)]


def build_direct_targets(
    df: pd.DataFrame, horizon_hours: int, base_column: str = "aqi"
) -> pd.DataFrame:
    """Add ``target_aqi_t_plus_1..H`` columns (future AQI), grouped by city."""
    if base_column not in df.columns:
        raise KeyError(f"Base target column '{base_column}' not present.")
    out = df.sort_values("timestamp").copy()
    grouped = (
        out.groupby("city_id", group_keys=False)[base_column] if "city_id" in out.columns else None
    )
    for h in range(1, horizon_hours + 1):
        col = f"{TARGET_PREFIX}{h}"
        if grouped is not None:
            out[col] = grouped.shift(-h)
        else:
            out[col] = out[base_column].shift(-h)
    return out


def trainable_rows(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """Rows with all horizon targets present (usable for supervised training)."""
    cols = target_columns(horizon_hours)
    present = [c for c in cols if c in df.columns]
    return df.dropna(subset=present) if present else df.iloc[0:0]
