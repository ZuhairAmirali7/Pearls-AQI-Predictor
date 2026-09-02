"""Baseline forecasters — cheap, transparent references the ML models must beat.

All four are *direct multi-horizon*: ``predict`` returns an ``(n, horizon)``
array. They reference feature columns by name and fall back gracefully to the
current AQI (or a stored global mean) when an expected column is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pearls_aqi.models.base import ForecastModel


def _col_or_fallback(X: pd.DataFrame, col: str, fallback: pd.Series | float) -> pd.Series:
    if col in X.columns:
        series = pd.to_numeric(X[col], errors="coerce")
        if isinstance(fallback, pd.Series):
            return series.fillna(fallback)
        return series.fillna(fallback)
    if isinstance(fallback, pd.Series):
        return fallback
    return pd.Series(fallback, index=X.index)


class PersistenceModel(ForecastModel):
    """Last-observed value carried forward to every horizon (naive persistence)."""

    model_type = "persistence"

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> PersistenceModel:
        self.global_mean_ = float(
            pd.to_numeric(X.get("aqi", pd.Series(dtype=float)), errors="coerce").mean() or 50.0
        )
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        current = _col_or_fallback(X, "aqi", self.global_mean_).to_numpy()
        return self._clip(np.tile(current.reshape(-1, 1), (1, self.horizon)))


class SeasonalNaiveModel(ForecastModel):
    """Value from the same hour ~24h ago (``aqi_lag_24h``), applied per horizon."""

    model_type = "seasonal_naive"

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> SeasonalNaiveModel:
        self.global_mean_ = float(
            pd.to_numeric(X.get("aqi", pd.Series(dtype=float)), errors="coerce").mean() or 50.0
        )
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        fallback = _col_or_fallback(X, "aqi", self.global_mean_)
        base = _col_or_fallback(X, "aqi_lag_24h", fallback).to_numpy()
        return self._clip(np.tile(base.reshape(-1, 1), (1, self.horizon)))


class RollingAverageModel(ForecastModel):
    """24-hour rolling mean of AQI (``aqi_roll_mean_24h``) carried to all horizons."""

    model_type = "rolling_average"

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> RollingAverageModel:
        self.global_mean_ = float(
            pd.to_numeric(X.get("aqi", pd.Series(dtype=float)), errors="coerce").mean() or 50.0
        )
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        fallback = _col_or_fallback(X, "aqi", self.global_mean_)
        base = _col_or_fallback(X, "aqi_roll_mean_24h", fallback).to_numpy()
        return self._clip(np.tile(base.reshape(-1, 1), (1, self.horizon)))


class HourOfDayModel(ForecastModel):
    """Historical average AQI by (hour-of-day, day-of-week) climatology.

    Fit builds the climatology from the current AQI observed at each
    (hour, day-of-week). Predict looks up the climatology for the *target* hour
    and day-of-week (advancing the calendar by the horizon), so each of the 72
    horizons gets its own seasonally-appropriate mean.
    """

    model_type = "hour_of_day"

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> HourOfDayModel:
        aqi = pd.to_numeric(X.get("aqi", pd.Series(dtype=float)), errors="coerce")
        self.global_mean_ = float(aqi.mean() or 50.0)
        self.table_: dict[tuple[int, int], float] = {}
        if {"hour", "day_of_week"} <= set(X.columns):
            grp = (
                pd.DataFrame(
                    {"hour": X["hour"].astype(int), "dow": X["day_of_week"].astype(int), "aqi": aqi}
                )
                .groupby(["hour", "dow"])["aqi"]
                .mean()
            )
            self.table_ = {(int(h), int(d)): float(v) for (h, d), v in grp.items()}
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        preds = np.full((n, self.horizon), self.global_mean_)
        if not ({"hour", "day_of_week"} <= set(X.columns)) or not self.table_:
            return self._clip(preds)
        hours = X["hour"].astype(int).to_numpy()
        dows = X["day_of_week"].astype(int).to_numpy()
        for h in range(1, self.horizon + 1):
            total = hours + h
            target_hour = total % 24
            target_dow = (dows + total // 24) % 7
            preds[:, h - 1] = [
                self.table_.get((int(th), int(td)), self.global_mean_)
                for th, td in zip(target_hour, target_dow, strict=False)
            ]
        return self._clip(preds)


BASELINE_MODELS: dict[str, type[ForecastModel]] = {
    PersistenceModel.model_type: PersistenceModel,
    SeasonalNaiveModel.model_type: SeasonalNaiveModel,
    RollingAverageModel.model_type: RollingAverageModel,
    HourOfDayModel.model_type: HourOfDayModel,
}
