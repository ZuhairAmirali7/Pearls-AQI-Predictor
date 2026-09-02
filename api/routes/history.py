"""Historical AQI / pollutant endpoint."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Query

from api.dependencies import get_feature_store, resolve_location
from api.schemas import HistoryResponse
from pearls_aqi.storage.base import FEATURES_GROUP
from pearls_aqi.utils.timeutils import parse_iso_or_none

router = APIRouter(prefix="/api/v1", tags=["history"])

_HISTORY_COLUMNS = [
    "timestamp",
    "aqi",
    "pm2_5",
    "pm10",
    "o3",
    "no2",
    "so2",
    "co",
    "dominant_pollutant",
]


@router.get("/history", response_model=HistoryResponse)
def history(
    city: str | None = Query(default=None),
    start: str | None = Query(default=None, description="ISO date/datetime (UTC)"),
    end: str | None = Query(default=None, description="ISO date/datetime (UTC)"),
    frequency: str = Query(default="hourly", pattern="^(hourly|daily)$"),
) -> HistoryResponse:
    location = resolve_location(city)
    fs = get_feature_store()
    start_dt = parse_iso_or_none(start)
    end_dt = parse_iso_or_none(end)
    df = fs.read_features(FEATURES_GROUP, start=start_dt, end=end_dt, city_id=location.city_id)

    if df is None or df.empty:
        return HistoryResponse(
            city=location.city, start=start, end=end, frequency=frequency, count=0, history=[]
        )
    cols = [c for c in _HISTORY_COLUMNS if c in df.columns]
    out = df[cols].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

    if frequency == "daily":
        numeric = out.select_dtypes("number").columns.tolist()
        out = out.set_index("timestamp")[numeric].resample("1D").mean().reset_index()
    out["timestamp"] = out["timestamp"].astype(str)
    records = out.round(2).to_dict(orient="records")
    return HistoryResponse(
        city=location.city,
        start=start,
        end=end,
        frequency=frequency,
        count=len(records),
        history=records,
    )
