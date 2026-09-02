"""72-hour AQI forecast endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_config, get_forecast_service, resolve_location
from api.rate_limit import rate_limiter
from api.schemas import AlertOut, ForecastPoint, ForecastResponse
from pearls_aqi.alerts import generate_alerts
from pearls_aqi.exceptions import ModelNotFoundError, PearlsError

router = APIRouter(prefix="/api/v1", tags=["forecast"])


@router.get("/forecast", response_model=ForecastResponse, dependencies=[Depends(rate_limiter)])
def forecast(
    city: str | None = Query(default=None),
    horizon: int | None = Query(default=None, ge=1, le=336),
    model_version: int | None = Query(default=None),
) -> ForecastResponse:
    location = resolve_location(city)
    service = get_forecast_service()
    try:
        result = service.generate(
            city=location.city, model_version=model_version, fetch_weather=False
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PearlsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    fc = result.forecast
    if horizon is not None:
        fc = fc[fc["horizon_hour"] <= horizon]

    alerts_cfg = get_config().alerts
    alerts = generate_alerts(fc, alerts_cfg, city=location.city)

    points = [
        ForecastPoint(
            timestamp=str(r.timestamp),
            horizon_hour=int(r.horizon_hour),
            predicted_aqi=round(float(r.predicted_aqi), 1),
            lower_bound=round(float(r.lower_bound), 1),
            upper_bound=round(float(r.upper_bound), 1),
            category=r.category,
            dominant_pollutant=r.dominant_pollutant,
        )
        for r in fc.itertuples()
    ]
    return ForecastResponse(
        city=result.city,
        generated_at=str(result.generated_at),
        model_name=result.model_name,
        model_version=result.model_version,
        forecast=points,
        alerts=[AlertOut(**{k: a.as_dict()[k] for k in AlertOut.model_fields}) for a in alerts],
    )
