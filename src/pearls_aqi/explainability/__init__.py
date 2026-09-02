"""Model explainability: SHAP when available, permutation/occlusion fallback."""

from __future__ import annotations

from pearls_aqi.explainability.explainer import (
    ExplanationResult,
    explain_forecast,
    global_feature_importance,
    local_feature_importance,
    plain_language_explanation,
)

__all__ = [
    "ExplanationResult",
    "explain_forecast",
    "global_feature_importance",
    "local_feature_importance",
    "plain_language_explanation",
]
