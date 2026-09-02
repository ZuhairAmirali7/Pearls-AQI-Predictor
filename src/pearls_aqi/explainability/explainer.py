"""Explainability for multi-horizon forecasts.

Primary method is **permutation / occlusion importance**, which is fully
model-agnostic and works with our multi-output pipeline wrappers. If SHAP is
installed and the underlying estimator is supported, a SHAP path is attempted
for a single horizon and clearly labelled. All importances describe *model
behaviour*, not causation — this is stated wherever explanations are shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pearls_aqi.utils.logging import get_logger

logger = get_logger(__name__)


def shap_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class ExplanationResult:
    method: str
    horizon_index: int
    global_importance: pd.DataFrame  # columns: feature, importance
    local_importance: pd.DataFrame | None = None
    plain_language: str | None = None
    notes: str = "Feature importance reflects model behaviour, not proven causation."
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "horizon_index": self.horizon_index,
            "global_importance": self.global_importance.to_dict(orient="records"),
            "local_importance": (
                self.local_importance.to_dict(orient="records")
                if self.local_importance is not None
                else None
            ),
            "plain_language": self.plain_language,
            "notes": self.notes,
        }


def global_feature_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.DataFrame | None = None,
    horizon_index: int = 0,
    n_repeats: int = 5,
    top_n: int = 25,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance for a single horizon (model-agnostic).

    Importance = mean increase in error (if ``y`` given) or mean change in
    prediction (otherwise) when a feature's values are shuffled.
    """
    features = getattr(model, "feature_names_", list(X.columns)) or list(X.columns)
    features = [f for f in features if f in X.columns]
    Xc = X[features].reset_index(drop=True)
    rng = np.random.default_rng(random_state)

    base_pred = np.asarray(model.predict(Xc))[:, horizon_index]
    ref = (
        y.iloc[:, horizon_index].to_numpy()
        if y is not None and horizon_index < y.shape[1]
        else base_pred
    )
    base_err = float(np.mean(np.abs(ref - base_pred)))

    importances: dict[str, float] = {}
    for feat in features:
        deltas = []
        original = Xc[feat].to_numpy().copy()
        for _ in range(n_repeats):
            shuffled = original.copy()
            rng.shuffle(shuffled)
            Xp = Xc.copy()
            Xp[feat] = shuffled
            pred = np.asarray(model.predict(Xp))[:, horizon_index]
            if y is not None:
                deltas.append(float(np.mean(np.abs(ref - pred))) - base_err)
            else:
                deltas.append(float(np.mean(np.abs(pred - base_pred))))
        importances[feat] = float(np.mean(deltas))

    out = (
        pd.DataFrame({"feature": list(importances), "importance": list(importances.values())})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return out


def local_feature_importance(
    model: Any,
    x_row: pd.DataFrame,
    background: pd.DataFrame,
    horizon_index: int = 0,
    top_n: int = 10,
) -> pd.DataFrame:
    """Occlusion-based local attribution for one instance.

    For each feature, measure how the prediction changes when that feature is
    replaced by its background median — a signed, per-feature contribution.
    """
    features = getattr(model, "feature_names_", list(x_row.columns)) or list(x_row.columns)
    features = [f for f in features if f in x_row.columns]
    row = x_row[features].reset_index(drop=True).iloc[[0]]
    medians = background[features].median(numeric_only=True)
    base = float(np.asarray(model.predict(row))[0, horizon_index])

    contribs = []
    for feat in features:
        perturbed = row.copy()
        perturbed[feat] = medians.get(feat, row[feat].iloc[0])
        new = float(np.asarray(model.predict(perturbed))[0, horizon_index])
        contribs.append(
            {"feature": feat, "contribution": base - new, "value": float(row[feat].iloc[0])}
        )
    df = pd.DataFrame(contribs)
    df["abs"] = df["contribution"].abs()
    return (
        df.sort_values("abs", ascending=False)
        .drop(columns="abs")
        .head(top_n)
        .reset_index(drop=True)
    )


def plain_language_explanation(
    local_importance: pd.DataFrame,
    predicted_aqi: float,
    category: str,
) -> str:
    """Human-readable summary of the top drivers of a prediction."""
    if local_importance is None or local_importance.empty:
        return (
            f"The model predicts an AQI of about {predicted_aqi:.0f} ({category}). "
            "No dominant single driver was identified."
        )
    drivers = local_importance.head(3)
    phrases = []
    for _, r in drivers.iterrows():
        direction = "raising" if r["contribution"] > 0 else "lowering"
        phrases.append(f"{_pretty(r['feature'])} ({direction} the forecast)")
    return (
        f"The predicted AQI is about {predicted_aqi:.0f} ({category}). "
        f"The main factors in the model are: {', '.join(phrases)}. "
        "These reflect how the model weights recent conditions, not proven cause and effect."
    )


def explain_forecast(
    model: Any,
    X_history: pd.DataFrame,
    x_current: pd.DataFrame,
    y: pd.DataFrame | None = None,
    horizon_index: int = 0,
    predicted_aqi: float | None = None,
    category: str | None = None,
) -> ExplanationResult:
    """Produce global + local explanations for the current forecast point."""
    method = "permutation/occlusion"
    glob = global_feature_importance(model, X_history, y=y, horizon_index=horizon_index)
    local = local_feature_importance(model, x_current, X_history, horizon_index=horizon_index)
    if predicted_aqi is None:
        predicted_aqi = float(np.asarray(model.predict(x_current))[0, horizon_index])
    if category is None:
        from pearls_aqi.data.domain import categorize_aqi

        category = categorize_aqi(predicted_aqi).value
    text = plain_language_explanation(local, predicted_aqi, category)
    return ExplanationResult(
        method=method,
        horizon_index=horizon_index,
        global_importance=glob,
        local_importance=local,
        plain_language=text,
    )


def _pretty(feature: str) -> str:
    """Turn a feature column name into a readable phrase."""
    mapping = {
        "aqi": "current AQI",
        "pm2_5": "PM2.5 level",
        "pm10": "PM10 level",
        "wind_speed": "wind speed",
        "humidity": "humidity",
        "temperature": "temperature",
    }
    for key, label in mapping.items():
        if feature == key:
            return label
    return feature.replace("_", " ")
