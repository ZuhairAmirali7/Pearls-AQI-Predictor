"""Dashboard data-access layer.

Works in two modes:
  * **Direct mode** (default): calls the ``pearls_aqi`` library in-process — the
    dashboard runs with no API server.
  * **HTTP mode**: when ``PEARLS_API_BASE_URL`` is set, calls the FastAPI backend.

Every function returns plain dicts / DataFrames so the pages stay backend-agnostic.
All functions degrade gracefully (return ``{"available": False, ...}`` or empty
frames) instead of raising, so the UI can show friendly messages.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
import requests

from pearls_aqi.config import load_config
from pearls_aqi.utils.env import load_dotenv

load_dotenv()


def api_base_url() -> str | None:
    url = os.getenv("PEARLS_API_BASE_URL")
    return url.rstrip("/") if url else None


def mode() -> str:
    return "api" if api_base_url() else "direct"


def get_config():
    return load_config()


def city_id_for(city: str | None) -> str:
    cfg = load_config()
    loc = cfg.active_location if not city else cfg.location_for_city(city)
    return loc.city_id


def list_cities() -> list[str]:
    cfg = load_config()
    return sorted({cfg.location.city, *cfg.cities.keys()})


# --- HTTP helpers -----------------------------------------------------------
def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    base = api_base_url()
    resp = requests.get(f"{base}{path}", params=params or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


# --- Forecast ---------------------------------------------------------------
def get_forecast(city: str | None = None) -> dict[str, Any]:
    """Return {available, forecast: DataFrame, meta: {...}, alerts: [...]}"""
    try:
        if mode() == "api":
            data = _get("/api/v1/forecast", {"city": city} if city else None)
            fc = pd.DataFrame(data["forecast"])
            if not fc.empty:
                fc["timestamp"] = pd.to_datetime(fc["timestamp"], utc=True)
            return {
                "available": True,
                "forecast": fc,
                "meta": {
                    "model_name": data.get("model_name"),
                    "model_version": data.get("model_version"),
                    "generated_at": data.get("generated_at"),
                    "city": data.get("city"),
                },
                "alerts": data.get("alerts", []),
                "weather": None,
            }
        # direct
        from pearls_aqi.alerts import generate_alerts
        from pearls_aqi.forecasting import ForecastService

        cfg = load_config()
        result = ForecastService(cfg).generate(city=city, fetch_weather=True)
        alerts = [
            a.as_dict() for a in generate_alerts(result.forecast, cfg.alerts, city=result.city)
        ]
        return {
            "available": True,
            "forecast": result.forecast,
            "meta": {
                "model_name": result.model_name,
                "model_version": result.model_version,
                "generated_at": result.generated_at.isoformat(),
                "city": result.city,
            },
            "alerts": alerts,
            "weather": result.weather_forecast,
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


# --- Current ----------------------------------------------------------------
def get_current(city: str | None = None) -> dict[str, Any]:
    try:
        if mode() == "api":
            return {"available": True, **_get("/api/v1/current", {"city": city} if city else None)}
        from pearls_aqi.data.domain import categorize_aqi
        from pearls_aqi.storage import get_feature_store
        from pearls_aqi.storage.base import FEATURES_GROUP

        cfg = load_config()
        cid = city_id_for(city)
        latest = get_feature_store(cfg).get_latest_features(cid, n=1, group=FEATURES_GROUP)
        if latest is None or latest.empty:
            return {"available": False, "reason": "No feature data. Run: make seed"}
        row = latest.iloc[0].to_dict()
        aqi = row.get("aqi")
        pollutants = {p: row.get(p) for p in ("pm2_5", "pm10", "o3", "no2", "so2", "co", "nh3")}
        weather = {
            k: row.get(k)
            for k in (
                "temperature",
                "feels_like",
                "humidity",
                "pressure",
                "wind_speed",
                "wind_direction",
                "cloud_cover",
                "visibility",
                "weather_condition",
            )
        }
        return {
            "available": True,
            "city": row.get("city"),
            "timestamp": pd.to_datetime(row.get("timestamp"), utc=True).isoformat(),
            "aqi": aqi,
            "category": categorize_aqi(aqi).value if aqi is not None else None,
            "dominant_pollutant": row.get("dominant_pollutant"),
            "pollutants": pollutants,
            "weather": weather,
            "data_source": row.get("data_source"),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


# --- History ----------------------------------------------------------------
def get_history(
    city: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    try:
        if mode() == "api":
            params: dict[str, Any] = {"city": city} if city else {}
            if start:
                params["start"] = start.isoformat()
            if end:
                params["end"] = end.isoformat()
            data = _get("/api/v1/history", params)
            df = pd.DataFrame(data.get("rows", data if isinstance(data, list) else []))
            if not df.empty and "timestamp" in df:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
        from pearls_aqi.storage import get_feature_store
        from pearls_aqi.storage.base import FEATURES_GROUP

        cfg = load_config()
        df = get_feature_store(cfg).read_features(
            FEATURES_GROUP, start=start, end=end, city_id=city_id_for(city)
        )
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# --- Model ------------------------------------------------------------------
def get_model_info(city: str | None = None) -> dict[str, Any]:
    try:
        if mode() == "api":
            return {"available": True, **_get("/api/v1/model", {"city": city} if city else None)}
        from pearls_aqi.registry import get_model_registry

        cfg = load_config()
        cid = city_id_for(city)
        _, meta = get_model_registry(cfg).load_latest_approved(
            model_name=f"aqi_forecast_{cid}", city_id=cid
        )
        return {"available": True, **meta.to_dict()}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def get_training_report(city: str | None = None) -> dict[str, Any] | None:
    from pearls_aqi.utils.io import read_json

    try:
        return read_json(f"artifacts/reports/{city_id_for(city)}/training_report_latest.json")
    except Exception:
        return None


def get_feature_run(city: str | None = None) -> dict[str, Any] | None:
    from pearls_aqi.utils.io import read_json

    try:
        return read_json(f"artifacts/runs/{city_id_for(city)}/feature_run_latest.json")
    except Exception:
        return None


# --- Monitoring / status ----------------------------------------------------
def get_system_status(city: str | None = None) -> dict[str, Any]:
    try:
        from pearls_aqi.monitoring import MonitoringService

        cfg = load_config()
        return MonitoringService(cfg).system_status(city_id_for(city))
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def get_model_performance(city: str | None = None) -> dict[str, Any]:
    try:
        from pearls_aqi.monitoring import MonitoringService

        cfg = load_config()
        return MonitoringService(cfg).model_performance(city_id_for(city))
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


# --- Explainability ---------------------------------------------------------
def get_explanation(city: str | None = None, horizon_index: int = 0) -> dict[str, Any]:
    try:
        if mode() == "api":
            params = {"horizon_index": horizon_index}
            if city:
                params["city"] = city
            return {"available": True, **_get("/api/v1/explanations", params)}
        from pearls_aqi.explainability import explain_forecast
        from pearls_aqi.registry import get_model_registry
        from pearls_aqi.storage import get_feature_store
        from pearls_aqi.storage.base import FEATURES_GROUP

        cfg = load_config()
        cid = city_id_for(city)
        model, meta = get_model_registry(cfg).load_latest_approved(
            model_name=f"aqi_forecast_{cid}", city_id=cid
        )
        rows = get_feature_store(cfg).read_features(FEATURES_GROUP, city_id=cid)
        if rows is None or rows.empty:
            return {"available": False, "reason": "No feature data."}
        feats = [c for c in meta.features if c in rows.columns]
        history = rows.tail(200)[feats]
        current = rows.tail(1)[feats]
        result = explain_forecast(model, history, current, horizon_index=horizon_index)
        return {"available": True, **result.as_dict()}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
