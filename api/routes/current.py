"""Current AQI + pollutants + weather for a city."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.dependencies import get_feature_store, resolve_location
from api.schemas import CurrentResponse, PollutantReadings
from pearls_aqi.data.domain import categorize_aqi
from pearls_aqi.data.schemas import WEATHER_COLUMNS
from pearls_aqi.storage.base import FEATURES_GROUP

router = APIRouter(prefix="/api/v1", tags=["current"])


@router.get("/current", response_model=CurrentResponse)
def current(city: str | None = Query(default=None, description="City name")) -> CurrentResponse:
    location = resolve_location(city)
    fs = get_feature_store()
    latest = fs.get_latest_features(location.city_id, n=1, group=FEATURES_GROUP)
    if latest is None or latest.empty:
        raise HTTPException(
            status_code=503,
            detail=f"No current data for {location.city}. Run the feature pipeline / seed data.",
        )
    row = latest.iloc[0]
    aqi = row.get("aqi")
    aqi_val = float(aqi) if aqi is not None and aqi == aqi else None
    weather = {c: _num(row.get(c)) for c in WEATHER_COLUMNS if c in latest.columns}
    return CurrentResponse(
        city=location.city,
        timestamp=str(row["timestamp"]),
        aqi=aqi_val,
        category=categorize_aqi(aqi_val).value if aqi_val is not None else None,
        dominant_pollutant=row.get("dominant_pollutant"),
        pollutants=PollutantReadings(
            pm2_5=_num(row.get("pm2_5")),
            pm10=_num(row.get("pm10")),
            o3=_num(row.get("o3")),
            no2=_num(row.get("no2")),
            so2=_num(row.get("so2")),
            co=_num(row.get("co")),
        ),
        weather=weather,
        data_source=row.get("data_source"),
    )


def _num(value) -> float | None:
    try:
        f = float(value)
        return f if f == f else None
    except (TypeError, ValueError):
        return None
