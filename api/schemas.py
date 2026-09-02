"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "pearls-aqi-predictor"
    version: str


class ReadyCheck(BaseModel):
    config_loaded: bool
    feature_store_reachable: bool
    model_available: bool
    schema_compatible: bool


class ReadyResponse(BaseModel):
    ready: bool
    checks: ReadyCheck
    detail: str | None = None


class PollutantReadings(BaseModel):
    pm2_5: float | None = None
    pm10: float | None = None
    o3: float | None = None
    no2: float | None = None
    so2: float | None = None
    co: float | None = None


class CurrentResponse(BaseModel):
    city: str
    timestamp: str
    aqi: float | None
    category: str | None
    dominant_pollutant: str | None
    pollutants: PollutantReadings
    weather: dict[str, Any]
    data_source: str | None = None


class ForecastPoint(BaseModel):
    timestamp: str
    horizon_hour: int
    predicted_aqi: float
    lower_bound: float
    upper_bound: float
    category: str
    dominant_pollutant: str | None = None


class AlertOut(BaseModel):
    type: str
    severity: str
    message: str
    peak_aqi: float
    start_time: str
    end_time: str


class ForecastResponse(BaseModel):
    city: str
    generated_at: str
    model_name: str
    model_version: int
    forecast: list[ForecastPoint]
    alerts: list[AlertOut] = Field(default_factory=list)


class HistoryPoint(BaseModel):
    timestamp: str
    aqi: float | None = None
    pm2_5: float | None = None
    pm10: float | None = None


class HistoryResponse(BaseModel):
    city: str
    start: str | None
    end: str | None
    frequency: str
    count: int
    history: list[dict[str, Any]]


class ModelInfoResponse(BaseModel):
    model_name: str
    version: int
    model_type: str
    status: str
    trained_at: str | None
    forecast_horizon: int
    city: str | None
    n_features: int

    model_config = {"protected_namespaces": ()}


class ModelMetricsResponse(BaseModel):
    model_name: str
    version: int
    metrics: dict[str, Any]

    model_config = {"protected_namespaces": ()}


class ExplanationResponse(BaseModel):
    method: str
    horizon_index: int
    plain_language: str | None
    global_importance: list[dict[str, Any]]
    local_importance: list[dict[str, Any]] | None
    notes: str


class PredictRequest(BaseModel):
    """Feature records for ad-hoc prediction (testing / alternative inputs)."""

    city: str | None = None
    records: list[dict[str, Any]] = Field(
        ..., description="List of feature dicts; unknown/missing features are imputed."
    )


class PredictResponse(BaseModel):
    model_name: str
    version: int
    horizon: int
    predictions: list[list[float]]

    model_config = {"protected_namespaces": ()}
