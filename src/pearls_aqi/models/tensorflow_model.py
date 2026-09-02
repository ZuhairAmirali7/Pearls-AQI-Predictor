"""TensorFlow/Keras feed-forward multi-horizon forecaster (OPTIONAL).

TensorFlow is imported lazily so importing this module never fails when the
package is absent. The training pipeline skips this model automatically if
TensorFlow is not installed, and the app never depends on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pearls_aqi.models.base import ForecastModel
from pearls_aqi.utils.io import ensure_dir, read_json, write_json
from pearls_aqi.utils.logging import get_logger

logger = get_logger(__name__)


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


class TensorFlowMLPModel(ForecastModel):
    """A small dense network with median imputation + standardisation."""

    model_type = "tensorflow"

    def __init__(
        self,
        horizon: int = 72,
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        epochs: int = 60,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        super().__init__(horizon=horizon)
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._keras: Any = None
        self._imputer: Any = None
        self._scaler: Any = None

    def _build_preprocess(self, X: pd.DataFrame) -> np.ndarray:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        arr = self._imputer.fit_transform(X)
        return self._scaler.fit_transform(arr)

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        arr = self._imputer.transform(X)
        return self._scaler.transform(arr)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> TensorFlowMLPModel:
        if not tensorflow_available():  # pragma: no cover
            raise RuntimeError("TensorFlow is not installed; cannot fit tensorflow model.")
        import tensorflow as tf

        tf.random.set_seed(self.random_state)
        self.feature_names_ = list(X.columns)
        X_arr = self._build_preprocess(X)
        y_arr = np.asarray(y, dtype="float32")

        model = tf.keras.Sequential(name="pearls_mlp")
        model.add(tf.keras.layers.Input(shape=(X_arr.shape[1],)))
        for units in self.hidden:
            model.add(tf.keras.layers.Dense(units, activation="relu"))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(self.dropout))
        model.add(tf.keras.layers.Dense(self.horizon))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="mae",
            metrics=["mse"],
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            )
        ]
        model.fit(
            X_arr,
            y_arr,
            validation_split=0.15,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
            shuffle=False,  # preserve temporal order
        )
        self._keras = model
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._keras is None:
            raise RuntimeError("tensorflow model is not fitted.")
        Xc = self._select(X)
        arr = self._transform(Xc)
        preds = np.asarray(self._keras.predict(arr, verbose=0)).reshape(len(X), -1)
        return self._clip(preds)

    # -- persistence (keras saved separately from preprocessing) ----------
    def save(self, directory: str | Path) -> Path:
        d = ensure_dir(directory)
        self._keras.save(d / "model_tf.keras")
        joblib.dump({"imputer": self._imputer, "scaler": self._scaler}, d / "preprocessing.joblib")
        write_json(
            {
                "horizon": self.horizon,
                "hidden": list(self.hidden),
                "dropout": self.dropout,
                "feature_names": self.feature_names_,
            },
            d / "tf_meta.json",
        )
        return d

    @classmethod
    def load(cls, directory: str | Path) -> TensorFlowMLPModel:
        import tensorflow as tf

        d = Path(directory)
        meta = read_json(d / "tf_meta.json")
        obj = cls(horizon=meta["horizon"], hidden=tuple(meta["hidden"]), dropout=meta["dropout"])
        obj.feature_names_ = meta["feature_names"]
        pre = joblib.load(d / "preprocessing.joblib")
        obj._imputer = pre["imputer"]
        obj._scaler = pre["scaler"]
        obj._keras = tf.keras.models.load_model(d / "model_tf.keras")
        obj.fitted_ = True
        return obj
