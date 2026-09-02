"""scikit-learn direct multi-horizon forecasters.

Preprocessing (median imputation, optional scaling) lives inside a ``Pipeline``
so it is **fit on training data only** — the whole pipeline is fit within each
CV fold and on the final training split, never on validation/test.

Multi-output handling:
  * Ridge, ElasticNet, RandomForest natively accept a 2-D target (one model,
    ``horizon`` outputs) — cheap.
  * HistGradientBoosting is single-output, so it is wrapped in
    ``MultiOutputRegressor`` (one estimator per horizon). To stay free-tier
    friendly it uses fixed, modest hyperparameters (no search).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pearls_aqi.models.base import ForecastModel
from pearls_aqi.utils.logging import get_logger

logger = get_logger(__name__)


class SklearnForecastModel(ForecastModel):
    """Base for pipeline-backed sklearn models."""

    model_type = "sklearn"
    scaling: bool = False
    supports_native_multioutput: bool = True

    def __init__(
        self,
        horizon: int = 72,
        tune: bool = False,
        n_iter: int = 10,
        cv_splits: int = 3,
        cv_gap: int = 24,
        random_state: int = 42,
        params: dict | None = None,
    ) -> None:
        super().__init__(horizon=horizon)
        self.tune = tune
        self.n_iter = n_iter
        self.cv_splits = cv_splits
        self.cv_gap = cv_gap
        self.random_state = random_state
        self.params = params or {}
        self.best_params_: dict = {}
        self.pipeline_: Pipeline | None = None

    # -- to override ------------------------------------------------------
    def _make_estimator(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def _param_distributions(self) -> dict:
        return {}

    # -- pipeline ---------------------------------------------------------
    def _build_pipeline(self) -> Pipeline:
        steps: list = [("impute", SimpleImputer(strategy="median"))]
        if self.scaling:
            steps.append(("scale", StandardScaler()))
        est = self._make_estimator()
        if not self.supports_native_multioutput:
            est = MultiOutputRegressor(est)
        steps.append(("model", est))
        return Pipeline(steps)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> SklearnForecastModel:
        self.feature_names_ = list(X.columns)
        y_arr = np.asarray(y)
        pipe = self._build_pipeline()
        dist = self._param_distributions() if self.tune else {}
        if dist:
            cv = TimeSeriesSplit(n_splits=self.cv_splits, gap=self.cv_gap)
            search = RandomizedSearchCV(
                pipe,
                dist,
                n_iter=self.n_iter,
                scoring="neg_mean_absolute_error",
                cv=cv,
                random_state=self.random_state,
                n_jobs=-1,
                error_score="raise",
            )
            search.fit(X, y_arr)
            self.pipeline_ = search.best_estimator_
            self.best_params_ = search.best_params_
            logger.info("%s best params: %s", self.model_type, search.best_params_)
        else:
            pipe.fit(X, y_arr)
            self.pipeline_ = pipe
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError(f"{self.model_type} is not fitted.")
        Xc = self._select(X)
        preds = np.asarray(self.pipeline_.predict(Xc)).reshape(len(X), -1)
        return self._clip(preds)


class RidgeModel(SklearnForecastModel):
    model_type = "ridge"
    scaling = True

    def _make_estimator(self):
        return Ridge(alpha=self.params.get("alpha", 1.0), random_state=self.random_state)

    def _param_distributions(self) -> dict:
        return {"model__alpha": loguniform(1e-2, 1e3)}


class ElasticNetModel(SklearnForecastModel):
    model_type = "elastic_net"
    scaling = True

    def _make_estimator(self):
        return ElasticNet(
            alpha=self.params.get("alpha", 0.1),
            l1_ratio=self.params.get("l1_ratio", 0.5),
            max_iter=5000,
            random_state=self.random_state,
        )

    def _param_distributions(self) -> dict:
        return {
            "model__alpha": loguniform(1e-3, 1e1),
            "model__l1_ratio": uniform(0.05, 0.9),
        }


class RandomForestModel(SklearnForecastModel):
    model_type = "random_forest"
    scaling = False

    def _make_estimator(self):
        return RandomForestRegressor(
            n_estimators=self.params.get("n_estimators", 200),
            max_depth=self.params.get("max_depth", None),
            min_samples_leaf=self.params.get("min_samples_leaf", 2),
            n_jobs=-1,
            random_state=self.random_state,
        )

    def _param_distributions(self) -> dict:
        return {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_leaf": [1, 2, 4],
        }


class HistGradientBoostingModel(SklearnForecastModel):
    model_type = "hist_gradient_boosting"
    scaling = False
    supports_native_multioutput = False  # wrapped in MultiOutputRegressor

    def _make_estimator(self):
        return HistGradientBoostingRegressor(
            learning_rate=self.params.get("learning_rate", 0.08),
            max_iter=self.params.get("max_iter", 200),
            max_leaf_nodes=self.params.get("max_leaf_nodes", 31),
            l2_regularization=self.params.get("l2_regularization", 0.0),
            random_state=self.random_state,
        )

    # No RandomizedSearch: 72 wrapped estimators make a search too costly for
    # the free tier. Fixed, sensible defaults are used instead (see model card).
    def _param_distributions(self) -> dict:
        return {}


SKLEARN_MODELS: dict[str, type[SklearnForecastModel]] = {
    RidgeModel.model_type: RidgeModel,
    ElasticNetModel.model_type: ElasticNetModel,
    RandomForestModel.model_type: RandomForestModel,
    HistGradientBoostingModel.model_type: HistGradientBoostingModel,
}
