"""Unit tests for model selection and promotion gating."""

from __future__ import annotations

import pytest

from pearls_aqi.evaluation.selection import evaluate_promotion, select_best_model


def test_select_best_by_validation_mae():
    results = [
        {"name": "a", "validation": {"mae": 10.0}},
        {"name": "b", "validation": {"mae": 8.0}},
        {"name": "c", "validation": {"mae": 12.0}},
    ]
    best = select_best_model(results)
    assert best["name"] == "b"
    assert "selection_rationale" in best


def test_select_tiebreak_by_hazard_mae():
    results = [
        {"name": "a", "validation": {"mae": 8.0}, "hazard": {"hazard_mae": 20.0}},
        {"name": "b", "validation": {"mae": 8.0}, "hazard": {"hazard_mae": 15.0}},
    ]
    assert select_best_model(results)["name"] == "b"


def test_select_raises_without_finite_scores():
    with pytest.raises(ValueError):
        select_best_model([{"name": "a", "validation": {"mae": float("nan")}}])


def test_promotion_bootstrap_no_incumbent():
    decision = evaluate_promotion({"validation": {"mae": 10.0}}, None)
    assert decision.promote is True


def test_promotion_rejects_insufficient_improvement():
    decision = evaluate_promotion(
        {"validation": {"mae": 9.95}},
        {"validation": {"mae": 10.0}},
        minimum_mae_improvement_percent=1.0,
    )
    assert decision.promote is False


def test_promotion_accepts_sufficient_improvement():
    decision = evaluate_promotion(
        {"validation": {"mae": 8.0}, "hazard": {"hazard_mae": 15.0}},
        {"validation": {"mae": 10.0}, "hazard": {"hazard_mae": 15.0}},
        minimum_mae_improvement_percent=1.0,
    )
    assert decision.promote is True


def test_promotion_rejects_hazard_degradation():
    decision = evaluate_promotion(
        {"validation": {"mae": 8.0}, "hazard": {"hazard_mae": 30.0}},
        {"validation": {"mae": 10.0}, "hazard": {"hazard_mae": 15.0}},
        maximum_hazard_mae_degradation_percent=2.0,
    )
    assert decision.promote is False


def test_promotion_rejects_failed_smoke():
    decision = evaluate_promotion(
        {"validation": {"mae": 8.0}},
        None,
        smoke_ok=False,
        require_smoke_test=True,
    )
    assert decision.promote is False
