"""Model explainability endpoint (global + local + plain language)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.dependencies import get_feature_store, get_registry, resolve_location
from api.schemas import ExplanationResponse
from pearls_aqi.data.domain import categorize_aqi
from pearls_aqi.exceptions import ModelNotFoundError
from pearls_aqi.explainability import explain_forecast
from pearls_aqi.storage.base import FEATURES_GROUP

router = APIRouter(prefix="/api/v1", tags=["explanations"])

# Small module cache keyed by (city_id, horizon_index).
_CACHE: dict[tuple[str, int], ExplanationResponse] = {}


@router.get("/explanations", response_model=ExplanationResponse)
def explanations(
    city: str | None = Query(default=None),
    horizon_index: int = Query(default=0, ge=0, description="0-based horizon hour index"),
) -> ExplanationResponse:
    location = resolve_location(city)
    cache_key = (location.city_id, horizon_index)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    registry = get_registry()
    try:
        model, meta = registry.load_latest_approved(
            model_name=f"aqi_forecast_{location.city_id}", city_id=location.city_id
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    fs = get_feature_store()
    history = fs.read_features(FEATURES_GROUP, city_id=location.city_id)
    if history is None or history.empty:
        raise HTTPException(
            status_code=503, detail="No feature history available for explanations."
        )

    features = [c for c in (meta.features or []) if c in history.columns]
    if not features:
        raise HTTPException(status_code=422, detail="Model features not present in current data.")

    sample = history.tail(200)
    x_current = history.tail(1)[features]
    horizon_index = min(horizon_index, max(0, model.horizon - 1))
    predicted = float(model.predict(x_current)[0, horizon_index])

    result = explain_forecast(
        model,
        sample[features],
        x_current,
        y=None,
        horizon_index=horizon_index,
        predicted_aqi=predicted,
        category=categorize_aqi(predicted).value,
    )
    response = ExplanationResponse(
        method=result.method,
        horizon_index=result.horizon_index,
        plain_language=result.plain_language,
        global_importance=result.global_importance.to_dict(orient="records"),
        local_importance=(
            result.local_importance.to_dict(orient="records")
            if result.local_importance is not None
            else None
        ),
        notes=result.notes,
    )
    _CACHE[cache_key] = response
    return response
