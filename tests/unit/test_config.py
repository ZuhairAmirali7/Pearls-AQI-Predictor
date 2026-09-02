"""Unit tests for configuration loading, overrides, and validation."""

from __future__ import annotations

import pytest

from pearls_aqi.config import load_config
from pearls_aqi.config.models import TrainingConfig
from pearls_aqi.exceptions import ConfigError


def test_loads_example_config():
    cfg = load_config()
    assert cfg.location.city
    assert cfg.forecast.horizon_hours == 72
    assert cfg.feature_store.backend in {"local", "hopsworks"}


def test_env_override_city(monkeypatch):
    monkeypatch.setenv("DEFAULT_CITY", "Lahore")
    monkeypatch.setenv("DEFAULT_COUNTRY", "Pakistan")
    monkeypatch.setenv("DEFAULT_LATITUDE", "31.5204")
    monkeypatch.setenv("DEFAULT_LONGITUDE", "74.3587")
    monkeypatch.setenv("DEFAULT_TIMEZONE", "Asia/Karachi")
    cfg = load_config()
    assert cfg.location.city == "Lahore"


def test_env_override_backend(monkeypatch):
    monkeypatch.setenv("FEATURE_STORE_BACKEND", "hopsworks")
    cfg = load_config()
    assert cfg.feature_store.backend == "hopsworks"


def test_invalid_fractions_raise():
    with pytest.raises(ValueError):
        TrainingConfig(train_fraction=0.8, validation_fraction=0.3, test_fraction=0.2)


def test_unknown_city_raises():
    cfg = load_config()
    with pytest.raises(ValueError):
        cfg.location_for_city("Atlantis")


def test_missing_config_file_raises():
    with pytest.raises(ConfigError):
        load_config("/nonexistent/config.yaml")
