"""Forecast evaluation metrics.

All functions accept 2-D arrays of shape ``(n_samples, horizon)`` (or 1-D) and
ignore NaN pairs. AQI-category and hazard-period breakdowns are included because
error near hazardous levels matters more than average error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pearls_aqi.data.domain import categorize_aqi


def _valid_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def rmse(y_true, y_pred) -> float:
    yt, yp = _valid_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt, yp = _valid_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def r2(y_true, y_pred) -> float:
    yt, yp = _valid_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def smape(y_true, y_pred) -> float:
    yt, yp = _valid_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    denom = np.where(denom == 0, np.nan, denom)
    return float(np.nanmean(np.abs(yt - yp) / denom) * 100)


def mape(y_true, y_pred) -> float:
    yt, yp = _valid_pair(y_true, y_pred)
    mask = yt != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)


def median_abs_error(y_true, y_pred) -> float:
    yt, yp = _valid_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.median(np.abs(yt - yp)))


def bias(y_true, y_pred) -> float:
    """Mean forecast error (pred - actual); positive => over-prediction."""
    yt, yp = _valid_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(yp - yt))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "median_ae": median_abs_error(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "n": int(_valid_pair(y_true, y_pred)[0].size),
    }


def metrics_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-horizon MAE/RMSE/bias — reveals forecast-skill degradation."""
    yt = np.atleast_2d(np.asarray(y_true, dtype=float))
    yp = np.atleast_2d(np.asarray(y_pred, dtype=float))
    rows = []
    for h in range(yt.shape[1]):
        rows.append(
            {
                "horizon_hour": h + 1,
                "mae": mae(yt[:, h], yp[:, h]),
                "rmse": rmse(yt[:, h], yp[:, h]),
                "bias": bias(yt[:, h], yp[:, h]),
            }
        )
    return pd.DataFrame(rows)


def metrics_by_category(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """MAE grouped by the true AQI category (error where it matters most)."""
    yt, yp = _valid_pair(y_true, y_pred)
    if yt.size == 0:
        return pd.DataFrame(columns=["category", "n", "mae"])
    cats = [categorize_aqi(v).value for v in yt]
    df = pd.DataFrame({"category": cats, "abs_err": np.abs(yt - yp)})
    out = df.groupby("category")["abs_err"].agg(["count", "mean"]).reset_index()
    return out.rename(columns={"count": "n", "mean": "mae"})


def hazard_metrics(y_true, y_pred, threshold: float = 151.0) -> dict[str, float]:
    """MAE/bias restricted to hazardous (true AQI >= threshold) samples."""
    yt, yp = _valid_pair(y_true, y_pred)
    mask = yt >= threshold
    if mask.sum() == 0:
        return {"hazard_mae": float("nan"), "hazard_bias": float("nan"), "hazard_n": 0}
    return {
        "hazard_mae": float(np.mean(np.abs(yt[mask] - yp[mask]))),
        "hazard_bias": float(np.mean(yp[mask] - yt[mask])),
        "hazard_n": int(mask.sum()),
    }


def category_confusion(y_true, y_pred) -> pd.DataFrame:
    """Confusion matrix of AQI categories from regression outputs."""
    from pearls_aqi.data.domain import AQICategory

    yt, yp = _valid_pair(y_true, y_pred)
    labels = [c.value for c in AQICategory]
    true_cats = pd.Categorical([categorize_aqi(v).value for v in yt], categories=labels)
    pred_cats = pd.Categorical([categorize_aqi(v).value for v in yp], categories=labels)
    return pd.crosstab(true_cats, pred_cats, dropna=False).reindex(
        index=labels, columns=labels, fill_value=0
    )
