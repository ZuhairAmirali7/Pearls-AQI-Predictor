"""Forecast-model interface and shared persistence.

Every model implements the same contract: ``fit(X, y) -> self`` and
``predict(X) -> ndarray[n, horizon]``. Predictions are direct multi-horizon
(one column per hour ``t+1 .. t+H``). Persistence defaults to joblib-pickling the
whole instance; the TensorFlow model overrides ``save``/``load``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pearls_aqi.utils.io import ensure_dir


class ForecastModel(ABC):
    """Abstract direct multi-horizon AQI forecaster."""

    #: Stable identifier persisted in metadata and used by the loader.
    model_type: str = "base"

    def __init__(self, horizon: int = 72) -> None:
        self.horizon = horizon
        self.feature_names_: list[str] = []
        self.fitted_ = False

    @property
    def name(self) -> str:
        return self.model_type

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> ForecastModel: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions of shape ``(len(X), horizon)``."""

    # -- helpers ----------------------------------------------------------
    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the model's feature columns in training order (0-filled if new)."""
        if not self.feature_names_:
            return X
        missing = [c for c in self.feature_names_ if c not in X.columns]
        Xc = X.copy()
        for c in missing:
            Xc[c] = np.nan
        return Xc[self.feature_names_]

    def _clip(self, preds: np.ndarray, lo: float = 0.0, hi: float = 500.0) -> np.ndarray:
        return np.clip(preds, lo, hi)

    # -- persistence ------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        d = ensure_dir(directory)
        path = d / "model.joblib"
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, directory: str | Path) -> ForecastModel:
        return joblib.load(Path(directory) / "model.joblib")


def load_generic(directory: str | Path) -> ForecastModel:
    """Load any joblib-pickled ForecastModel from a version directory."""
    return joblib.load(Path(directory) / "model.joblib")
