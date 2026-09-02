"""Model zoo: baselines, sklearn regressors, and an optional TF network.

``build_model`` constructs a model by name from config; ``load_model`` restores a
persisted model by ``model_type`` (used by the registry).
"""

from __future__ import annotations

from pathlib import Path

from pearls_aqi.config.models import TrainingConfig
from pearls_aqi.models.base import ForecastModel, load_generic
from pearls_aqi.models.baselines import BASELINE_MODELS
from pearls_aqi.models.sklearn_models import SKLEARN_MODELS

# Names that require optional TensorFlow.
TENSORFLOW_MODELS = {"tensorflow"}

ALL_MODEL_NAMES = sorted({*BASELINE_MODELS, *SKLEARN_MODELS, *TENSORFLOW_MODELS})

__all__ = [
    "ALL_MODEL_NAMES",
    "TENSORFLOW_MODELS",
    "ForecastModel",
    "build_model",
    "load_model",
]


def build_model(
    name: str,
    horizon: int,
    training: TrainingConfig | None = None,
    tune: bool = False,
) -> ForecastModel:
    """Instantiate a model by name."""
    if name in BASELINE_MODELS:
        return BASELINE_MODELS[name](horizon=horizon)
    if name in SKLEARN_MODELS:
        t = training or TrainingConfig()
        return SKLEARN_MODELS[name](
            horizon=horizon,
            tune=tune,
            n_iter=t.random_search_iterations,
            cv_splits=t.cv_splits,
            cv_gap=t.cv_gap_hours,
            random_state=t.random_state,
        )
    if name in TENSORFLOW_MODELS:
        from pearls_aqi.models.tensorflow_model import TensorFlowMLPModel

        return TensorFlowMLPModel(
            horizon=horizon, random_state=(training or TrainingConfig()).random_state
        )
    raise ValueError(f"Unknown model name: {name!r}. Known: {ALL_MODEL_NAMES}")


def load_model(model_type: str, directory: str | Path) -> ForecastModel:
    """Restore a persisted model from a version directory."""
    if model_type in TENSORFLOW_MODELS:
        from pearls_aqi.models.tensorflow_model import TensorFlowMLPModel

        return TensorFlowMLPModel.load(directory)
    return load_generic(directory)


def is_available(name: str) -> bool:
    """Whether a model can be built in this environment (TF may be missing)."""
    if name in TENSORFLOW_MODELS:
        from pearls_aqi.models.tensorflow_model import tensorflow_available

        return tensorflow_available()
    return name in ALL_MODEL_NAMES
