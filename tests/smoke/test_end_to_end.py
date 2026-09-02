"""Smoke test: full offline pipeline on fixture data in a temp directory.

Confirms the whole system works end-to-end without network or credentials:
seed → train → load approved model → generate a forecast → evaluate alerts.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from pearls_aqi.alerts import generate_alerts
from pearls_aqi.api_clients.sample_provider import (
    SampleAirQualityProvider,
    SampleWeatherProvider,
)
from pearls_aqi.forecasting import ForecastService
from pearls_aqi.pipelines import FeaturePipeline, TrainingPipeline
from pearls_aqi.registry.local import LocalModelRegistry
from pearls_aqi.storage.local import LocalParquetFeatureStore
from pearls_aqi.utils.timeutils import now_utc, utc_floor_hour

pytestmark = pytest.mark.smoke


def test_full_pipeline_smoke(sample_config, location, tmp_path):
    store = LocalParquetFeatureStore(tmp_path / "fs")
    registry = LocalModelRegistry(tmp_path / "models")

    # 1. Seed ~45 days of sample features.
    fp = FeaturePipeline(sample_config, SampleAirQualityProvider(), SampleWeatherProvider(), store)
    end = utc_floor_hour(now_utc())
    fp.run(location, end - timedelta(days=45), end)

    # 2. Train a small model set (horizon 6) and approve the winner.
    tp = TrainingPipeline(sample_config, store, registry)
    report = tp.run(
        location, horizon=6, models=["persistence", "ridge", "random_forest"], tune=False
    )
    assert report.registered_status == "approved"

    # 3. Load the approved model and generate a forecast.
    service = ForecastService(
        sample_config,
        feature_store=store,
        registry=registry,
        weather_provider=SampleWeatherProvider(),
    )
    result = service.generate(city=location.city)
    assert len(result.forecast) == 6
    assert np.all(np.isfinite(result.forecast["predicted_aqi"]))

    # 4. Alert evaluation runs without error.
    alerts = generate_alerts(result.forecast, sample_config.alerts, city=location.city)
    assert isinstance(alerts, list)
