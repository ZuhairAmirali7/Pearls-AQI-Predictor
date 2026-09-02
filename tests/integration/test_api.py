"""Integration tests for the FastAPI backend.

The route modules resolve their services through ``api.dependencies`` getters
imported by name; we monkeypatch those names in each route module to point at
the fixture-backed store + registry (session fixtures from conftest). This keeps
the tests fully offline and avoids touching the real default artifacts.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pearls_aqi.config import load_config
from pearls_aqi.forecasting import ForecastService

pytestmark = pytest.mark.integration


@pytest.fixture
def client(monkeypatch, seeded_store, trained_registry):
    registry, _report = trained_registry
    config = load_config()
    forecast_service = ForecastService(config, feature_store=seeded_store, registry=registry)

    # Patch the dependency getters as imported into each route module.
    patches = {
        "api.routes.forecast.get_forecast_service": lambda: forecast_service,
        "api.routes.forecast.get_config": lambda: config,
        "api.routes.current.get_feature_store": lambda: seeded_store,
        "api.routes.history.get_feature_store": lambda: seeded_store,
        "api.routes.model.get_registry": lambda: registry,
        "api.routes.predict.get_registry": lambda: registry,
        "api.routes.explanations.get_feature_store": lambda: seeded_store,
        "api.routes.explanations.get_registry": lambda: registry,
        "api.routes.health.get_feature_store": lambda: seeded_store,
        "api.routes.health.get_registry": lambda: registry,
        "api.routes.health.get_config": lambda: config,
    }
    for target, value in patches.items():
        monkeypatch.setattr(target, value)

    from api.main import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert "checks" in body
    assert body["checks"]["model_available"] is True


def test_forecast_shape(client, trained_registry):
    _registry, report = trained_registry
    r = client.get("/api/v1/forecast")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"].startswith("aqi_forecast_")
    assert len(body["forecast"]) == report.horizon
    point = body["forecast"][0]
    for key in (
        "timestamp",
        "horizon_hour",
        "predicted_aqi",
        "lower_bound",
        "upper_bound",
        "category",
    ):
        assert key in point
    assert 0 <= point["predicted_aqi"] <= 500
    assert point["lower_bound"] <= point["predicted_aqi"] <= point["upper_bound"]


def test_forecast_horizon_param(client):
    r = client.get("/api/v1/forecast", params={"horizon": 3})
    assert r.status_code == 200
    assert len(r.json()["forecast"]) == 3


def test_current(client):
    r = client.get("/api/v1/current")
    assert r.status_code == 200
    body = r.json()
    assert "aqi" in body and "pollutants" in body and "weather" in body


def test_model_and_metrics(client):
    r = client.get("/api/v1/model")
    assert r.status_code == 200
    assert r.json()["forecast_horizon"] >= 1
    m = client.get("/api/v1/model/metrics")
    assert m.status_code == 200
    assert "metrics" in m.json()


def test_history(client):
    r = client.get("/api/v1/history", params={"frequency": "daily"})
    assert r.status_code == 200
    assert r.json()["frequency"] == "daily"


def test_predict_and_schema_mismatch(client, seeded_store, trained_registry):
    registry, _ = trained_registry
    _model, meta = registry.load_latest_approved(
        model_name="aqi_forecast_karachi_pakistan", city_id="karachi_pakistan"
    )
    from pearls_aqi.storage.base import FEATURES_GROUP

    row = seeded_store.get_latest_features("karachi_pakistan", n=1, group=FEATURES_GROUP)
    rec = {
        k: (None if row[k].isna().iloc[0] else float(row[k].iloc[0]))
        for k in meta.features
        if k in row.columns
    }
    r = client.post("/api/v1/predict", json={"records": [rec]})
    assert r.status_code == 200
    assert len(r.json()["predictions"][0]) == meta.forecast_horizon

    bad = client.post("/api/v1/predict", json={"records": [{"foo": 1.0}]})
    assert bad.status_code == 422
