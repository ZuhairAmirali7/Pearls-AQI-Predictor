"""Liveness and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from api.dependencies import (
    get_config,
    get_feature_store,
    get_registry,
    resolve_location,
)
from api.schemas import HealthResponse, ReadyCheck, ReadyResponse
from pearls_aqi import __version__

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: the process is up."""
    return HealthResponse(version=__version__)


@router.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    """Readiness: config, feature store, model, and schema are all usable."""
    config_loaded = feature_store_ok = model_ok = schema_ok = False
    detail = None
    try:
        get_config()
        config_loaded = True
        location = resolve_location(None)
        fs = get_feature_store()
        latest = fs.get_latest_features(location.city_id, n=1)
        feature_store_ok = latest is not None
        registry = get_registry()
        model, _meta = registry.load_latest_approved(
            model_name=f"aqi_forecast_{location.city_id}", city_id=location.city_id
        )
        model_ok = True
        names = getattr(model, "feature_names_", []) or []
        if latest is not None and not latest.empty and names:
            overlap = sum(1 for c in names if c in latest.columns) / len(names)
            schema_ok = overlap >= 0.8
        else:
            schema_ok = model_ok
    except Exception as exc:
        detail = str(exc)

    checks = ReadyCheck(
        config_loaded=config_loaded,
        feature_store_reachable=feature_store_ok,
        model_available=model_ok,
        schema_compatible=schema_ok,
    )
    ready_flag = all([config_loaded, feature_store_ok, model_ok, schema_ok])
    return ReadyResponse(ready=ready_flag, checks=checks, detail=detail)
