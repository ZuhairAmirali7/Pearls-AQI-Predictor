"""Prediction service — turns the approved model + latest features into a forecast.

Strategy: **direct multi-horizon**. The most recent engineered feature row (time
``t``, containing all lag/rolling/calendar features) is fed to the model, which
outputs AQI for ``t+1 .. t+H`` in one shot. The service also fetches the future
hourly weather forecast (best-effort) and returns it as context; see
``docs/limitations.md`` for how future weather relates to the model inputs.

Prediction-interval widths use per-horizon validation MAE stored in the model
metadata when available, else a horizon-scaled heuristic — labelled as
approximate, not calibrated coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from pearls_aqi.config.models import AppConfig
from pearls_aqi.data.domain import Location, categorize_aqi
from pearls_aqi.exceptions import PredictionError, SchemaMismatchError
from pearls_aqi.registry.base import ModelMetadata
from pearls_aqi.storage.base import PREDICTIONS_GROUP
from pearls_aqi.utils.logging import get_logger
from pearls_aqi.utils.timeutils import ensure_utc, now_utc

logger = get_logger(__name__)

_Z_80 = 1.2816  # ~80% interval under a normal approximation


@dataclass
class ForecastResult:
    city: str
    generated_at: datetime
    model_name: str
    model_version: int
    location: Location
    forecast: (
        pd.DataFrame
    )  # columns: timestamp, horizon_hour, predicted_aqi, lower/upper, category, dominant_pollutant
    weather_forecast: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "city": self.city,
            "generated_at": ensure_utc(self.generated_at).isoformat(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "forecast": [
                {
                    "timestamp": ensure_utc(row.timestamp).isoformat(),
                    "horizon_hour": int(row.horizon_hour),
                    "predicted_aqi": round(float(row.predicted_aqi), 1),
                    "lower_bound": round(float(row.lower_bound), 1),
                    "upper_bound": round(float(row.upper_bound), 1),
                    "category": row.category,
                    "dominant_pollutant": row.dominant_pollutant,
                }
                for row in self.forecast.itertuples()
            ],
        }


class ForecastService:
    def __init__(
        self,
        config: AppConfig,
        feature_store: Any = None,
        registry: Any = None,
        weather_provider: Any = None,
    ) -> None:
        self.config = config
        self._feature_store = feature_store
        self._registry = registry
        self._weather_provider = weather_provider

    # Lazily build collaborators so importing this module is cheap and tests can
    # inject fakes.
    @property
    def feature_store(self):
        if self._feature_store is None:
            from pearls_aqi.storage import get_feature_store

            self._feature_store = get_feature_store(self.config)
        return self._feature_store

    @property
    def registry(self):
        if self._registry is None:
            from pearls_aqi.registry import get_model_registry

            self._registry = get_model_registry(self.config)
        return self._registry

    def _load_model(self, model_name: str | None, version: int | None, city_id: str):
        if model_name and version:
            return self.registry.load(model_name, version)
        return self.registry.load_latest_approved(model_name=model_name, city_id=city_id)

    def generate(
        self,
        city: str | None = None,
        model_name: str | None = None,
        model_version: int | None = None,
        fetch_weather: bool = True,
    ) -> ForecastResult:
        location = (
            self.config.active_location if city is None else self.config.location_for_city(city)
        )
        city_id = location.city_id

        model, meta = self._load_model(model_name, model_version, city_id)
        latest = self.feature_store.get_latest_features(city_id, n=1)
        if latest is None or latest.empty:
            raise PredictionError(
                f"No features available for {location.city}. Run the feature "
                "pipeline (or seed sample data) first."
            )

        self._check_schema(model, latest)
        preds = model.predict(latest)[0]  # (horizon,)
        horizon = int(getattr(model, "horizon", len(preds)))
        preds = np.clip(preds, self.config.forecast.min_aqi, self.config.forecast.max_aqi)

        t0 = ensure_utc(pd.to_datetime(latest["timestamp"].iloc[0]).to_pydatetime())
        timestamps = [t0 + pd.Timedelta(hours=h) for h in range(1, horizon + 1)]
        errors = self._per_horizon_error(meta, horizon)
        dominant = latest.get("dominant_pollutant")
        dominant_val = dominant.iloc[0] if dominant is not None and len(dominant) else None

        forecast = pd.DataFrame(
            {
                "timestamp": timestamps,
                "horizon_hour": list(range(1, horizon + 1)),
                "predicted_aqi": preds,
                "lower_bound": np.clip(
                    preds - _Z_80 * errors,
                    self.config.forecast.min_aqi,
                    self.config.forecast.max_aqi,
                ),
                "upper_bound": np.clip(
                    preds + _Z_80 * errors,
                    self.config.forecast.min_aqi,
                    self.config.forecast.max_aqi,
                ),
                "category": [categorize_aqi(v).value for v in preds],
                "dominant_pollutant": dominant_val,
            }
        )

        weather_forecast = None
        if fetch_weather:
            weather_forecast = self._safe_weather_forecast(location, horizon)

        result = ForecastResult(
            city=location.city,
            generated_at=now_utc(),
            model_name=meta.model_name,
            model_version=meta.version,
            location=location,
            forecast=forecast,
            weather_forecast=weather_forecast,
            metadata={"feature_time": t0.isoformat(), "model_type": meta.model_type},
        )
        self._store_predictions(result, city_id, location)
        logger.info(
            "Generated %dh forecast for %s using %s v%d (peak AQI %.0f).",
            horizon,
            location.city,
            meta.model_name,
            meta.version,
            float(np.max(preds)),
        )
        return result

    # -- helpers ----------------------------------------------------------
    def _check_schema(self, model, latest: pd.DataFrame) -> None:
        names = getattr(model, "feature_names_", None)
        if not names:
            return
        available = set(latest.columns)
        overlap = sum(1 for c in names if c in available) / len(names)
        if overlap < 0.8:
            raise SchemaMismatchError(
                f"Feature schema mismatch: only {overlap:.0%} of the model's "
                f"{len(names)} features are present in the latest data. Retrain "
                "the model or rebuild features."
            )

    def _per_horizon_error(self, meta: ModelMetadata, horizon: int) -> np.ndarray:
        by_h = meta.metrics.get("by_horizon_mae") if meta.metrics else None
        if isinstance(by_h, list) and len(by_h) >= horizon:
            return np.array(by_h[:horizon], dtype=float)
        base = float(meta.metrics.get("validation_mae", 8.0)) if meta.metrics else 8.0
        return base * (1.0 + 0.03 * np.arange(1, horizon + 1))

    def _safe_weather_forecast(self, location: Location, horizon: int) -> pd.DataFrame | None:
        try:
            provider = self._weather_provider
            if provider is None:
                from pearls_aqi.api_clients import get_weather_provider

                provider = get_weather_provider(self.config)
            return provider.fetch_forecast(location, horizon)
        except Exception as exc:
            logger.warning("Future weather forecast unavailable: %s", exc)
            return None

    def _store_predictions(self, result: ForecastResult, city_id: str, location: Location) -> None:
        try:
            df = result.forecast.copy()
            df["city_id"] = city_id
            df["city"] = location.city
            df["model_name"] = result.model_name
            df["model_version"] = result.model_version
            df["generated_at"] = ensure_utc(result.generated_at)
            self.feature_store.write_features(df, group=PREDICTIONS_GROUP)
        except Exception as exc:
            logger.warning("Could not persist predictions: %s", exc)
