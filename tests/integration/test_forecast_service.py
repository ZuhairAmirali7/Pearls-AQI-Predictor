"""Integration test: end-to-end forecast generation from the approved model."""

from __future__ import annotations

import pytest

from pearls_aqi.api_clients.sample_provider import SampleWeatherProvider
from pearls_aqi.forecasting import ForecastService

pytestmark = pytest.mark.integration


def test_forecast_service_generates_valid_forecast(
    sample_config, location, seeded_store, trained_registry
):
    registry, report = trained_registry
    service = ForecastService(
        sample_config,
        feature_store=seeded_store,
        registry=registry,
        weather_provider=SampleWeatherProvider(),
    )
    result = service.generate(city=location.city)

    fc = result.forecast
    assert len(fc) == report.horizon
    # Interval ordering and clipping.
    assert (fc["lower_bound"] <= fc["predicted_aqi"]).all()
    assert (fc["predicted_aqi"] <= fc["upper_bound"]).all()
    assert fc["predicted_aqi"].between(0, 500).all()
    # Categories assigned.
    assert fc["category"].notna().all()
    # Horizon hours are 1..H.
    assert list(fc["horizon_hour"]) == list(range(1, report.horizon + 1))


def test_forecast_result_dict_schema(sample_config, location, seeded_store, trained_registry):
    registry, _ = trained_registry
    service = ForecastService(
        sample_config,
        feature_store=seeded_store,
        registry=registry,
        weather_provider=SampleWeatherProvider(),
    )
    payload = service.generate(city=location.city).as_dict()
    assert {"city", "generated_at", "model_name", "model_version", "forecast"} <= set(payload)
    first = payload["forecast"][0]
    assert {
        "timestamp",
        "horizon_hour",
        "predicted_aqi",
        "lower_bound",
        "upper_bound",
        "category",
        "dominant_pollutant",
    } <= set(first)
