"""Contract tests: prediction output + registered model metadata schemas."""

from __future__ import annotations

import pytest

from pearls_aqi.api_clients.sample_provider import SampleWeatherProvider
from pearls_aqi.forecasting import ForecastService

pytestmark = pytest.mark.contract

_REQUIRED_FORECAST_KEYS = {
    "timestamp",
    "horizon_hour",
    "predicted_aqi",
    "lower_bound",
    "upper_bound",
    "category",
    "dominant_pollutant",
}
_REQUIRED_METADATA_FIELDS = {
    "model_name",
    "version",
    "model_type",
    "status",
    "created_at",
    "features",
    "target",
    "forecast_horizon",
    "city_id",
    "metrics",
}


def test_prediction_response_contract(sample_config, location, seeded_store, trained_registry):
    registry, _ = trained_registry
    service = ForecastService(
        sample_config,
        feature_store=seeded_store,
        registry=registry,
        weather_provider=SampleWeatherProvider(),
    )
    payload = service.generate(city=location.city).as_dict()
    assert {"city", "generated_at", "model_name", "model_version", "forecast"} <= set(payload)
    assert payload["forecast"], "forecast must be non-empty"
    for point in payload["forecast"]:
        assert set(point) >= _REQUIRED_FORECAST_KEYS


def test_registered_metadata_contract(trained_registry, location):
    registry, report = trained_registry
    _, meta = registry.load_latest_approved(model_name=report.model_name, city_id=location.city_id)
    assert set(meta.to_dict()) >= _REQUIRED_METADATA_FIELDS
    # Reproducibility fields captured.
    assert meta.python_version
    assert isinstance(meta.dependencies, dict) and meta.dependencies
