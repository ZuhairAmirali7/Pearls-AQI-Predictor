"""Integration test: training pipeline (uses the trained_registry fixture)."""

from __future__ import annotations

import math

import pytest

from pearls_aqi.registry.base import ModelStatus

pytestmark = pytest.mark.integration


def test_training_registers_and_approves(trained_registry):
    _registry, report = trained_registry
    assert report.registered_version >= 1
    assert report.registered_status == ModelStatus.APPROVED.value
    assert report.selected is not None


def test_training_compares_at_least_four_models(trained_registry):
    _, report = trained_registry
    successful = [r for r in report.results if not r.error]
    assert len(successful) >= 4


def test_training_metrics_are_finite_and_complete(trained_registry):
    _, report = trained_registry
    winner = next(r for r in report.results if r.name == report.selected)
    for key in ("mae", "rmse", "r2"):
        assert key in winner.validation
        assert math.isfinite(winner.validation[key])


def test_approved_model_loads_from_registry(trained_registry, location):
    registry, report = trained_registry
    _model, meta = registry.load_latest_approved(
        model_name=report.model_name, city_id=location.city_id
    )
    assert meta.status == ModelStatus.APPROVED.value
    assert meta.forecast_horizon == report.horizon
    assert len(meta.features) > 0
