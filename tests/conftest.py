"""Shared pytest fixtures.

Expensive fixtures (seeding a feature store, training a small model) are
session-scoped and use ``tmp_path_factory`` so they run once for the whole suite.
Everything is offline: the sample providers generate deterministic synthetic
data — no network is touched.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from pearls_aqi.api_clients.sample_provider import (
    SampleAirQualityProvider,
    SampleWeatherProvider,
)
from pearls_aqi.config import load_config
from pearls_aqi.data.sample import generate_sample_observations
from pearls_aqi.pipelines import FeaturePipeline, TrainingPipeline
from pearls_aqi.registry.local import LocalModelRegistry
from pearls_aqi.storage.local import LocalParquetFeatureStore
from pearls_aqi.utils.timeutils import now_utc, utc_floor_hour


@pytest.fixture(scope="session")
def sample_config():
    return load_config()


@pytest.fixture(scope="session")
def location(sample_config):
    return sample_config.active_location


@pytest.fixture
def sample_observations(location):
    end = utc_floor_hour(now_utc())
    start = end - timedelta(days=30)
    return generate_sample_observations(location, start, end)


@pytest.fixture
def tmp_feature_store(tmp_path):
    return LocalParquetFeatureStore(tmp_path / "fs")


@pytest.fixture
def tmp_registry(tmp_path):
    return LocalModelRegistry(tmp_path / "models")


@pytest.fixture(scope="session")
def seeded_store(sample_config, location, tmp_path_factory):
    store = LocalParquetFeatureStore(tmp_path_factory.mktemp("fs"))
    pipeline = FeaturePipeline(
        sample_config, SampleAirQualityProvider(), SampleWeatherProvider(), store
    )
    end = utc_floor_hour(now_utc())
    start = end - timedelta(days=60)
    pipeline.run(location, start, end)
    return store


@pytest.fixture(scope="session")
def trained_registry(sample_config, location, seeded_store, tmp_path_factory):
    registry = LocalModelRegistry(tmp_path_factory.mktemp("models"))
    pipeline = TrainingPipeline(sample_config, seeded_store, registry)
    report = pipeline.run(
        location,
        horizon=6,
        models=["persistence", "seasonal_naive", "ridge", "random_forest"],
        tune=False,
        # Isolate the report so tests never overwrite the real default report.
        report_dir=str(tmp_path_factory.mktemp("reports")),
    )
    return registry, report
