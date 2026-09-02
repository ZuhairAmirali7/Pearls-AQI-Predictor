"""Approved-model metadata and metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.dependencies import get_registry, resolve_location
from api.schemas import ModelInfoResponse, ModelMetricsResponse
from pearls_aqi.exceptions import ModelNotFoundError


def _load_meta(city: str | None):
    location = resolve_location(city)
    registry = get_registry()
    try:
        _model, meta = registry.load_latest_approved(
            model_name=f"aqi_forecast_{location.city_id}", city_id=location.city_id
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return meta


router = APIRouter(prefix="/api/v1", tags=["model"])


@router.get("/model", response_model=ModelInfoResponse)
def model_info(city: str | None = Query(default=None)) -> ModelInfoResponse:
    meta = _load_meta(city)
    return ModelInfoResponse(
        model_name=meta.model_name,
        version=meta.version,
        model_type=meta.model_type,
        status=meta.status,
        trained_at=meta.created_at,
        forecast_horizon=meta.forecast_horizon,
        city=meta.city,
        n_features=len(meta.features),
    )


@router.get("/model/metrics", response_model=ModelMetricsResponse)
def model_metrics(city: str | None = Query(default=None)) -> ModelMetricsResponse:
    meta = _load_meta(city)
    return ModelMetricsResponse(
        model_name=meta.model_name, version=meta.version, metrics=meta.metrics
    )
