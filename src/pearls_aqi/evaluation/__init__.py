"""Evaluation: metrics, chronological splits, and model selection."""

from __future__ import annotations

from pearls_aqi.evaluation.metrics import (
    hazard_metrics,
    metrics_by_category,
    metrics_by_horizon,
    regression_metrics,
)
from pearls_aqi.evaluation.selection import (
    PromotionDecision,
    evaluate_promotion,
    select_best_model,
)
from pearls_aqi.evaluation.splits import chronological_split

__all__ = [
    "PromotionDecision",
    "chronological_split",
    "evaluate_promotion",
    "hazard_metrics",
    "metrics_by_category",
    "metrics_by_horizon",
    "regression_metrics",
    "select_best_model",
]
