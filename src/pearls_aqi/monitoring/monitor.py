"""Monitoring service: data freshness, live model error, drift, system status.

Results are plain dicts (JSON-serialisable) so they can be surfaced by the API
and dashboard and optionally written to the ``model_monitoring`` feature group.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from pearls_aqi.config.models import AppConfig
from pearls_aqi.evaluation.metrics import metrics_by_category, regression_metrics
from pearls_aqi.storage.base import FEATURES_GROUP, PREDICTIONS_GROUP
from pearls_aqi.utils.io import read_json
from pearls_aqi.utils.logging import get_logger
from pearls_aqi.utils.timeutils import ensure_utc, now_utc

logger = get_logger(__name__)


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI between two distributions. >0.2 conventionally signals drift."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if expected.size == 0 or actual.size == 0:
        return float("nan")
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(expected, quantiles))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_frac = np.clip(e_counts / e_counts.sum(), 1e-6, None)
    a_frac = np.clip(a_counts / a_counts.sum(), 1e-6, None)
    return float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))


class MonitoringService:
    def __init__(self, config: AppConfig, feature_store: Any = None) -> None:
        self.config = config
        self._fs = feature_store

    @property
    def feature_store(self):
        if self._fs is None:
            from pearls_aqi.storage import get_feature_store

            self._fs = get_feature_store(self.config)
        return self._fs

    # -- data monitoring --------------------------------------------------
    def data_health(self, city_id: str) -> dict[str, Any]:
        df = self.feature_store.read_features(FEATURES_GROUP, city_id=city_id)
        if df is None or df.empty:
            return {"available": False, "reason": "no feature data"}
        ts = pd.to_datetime(df["timestamp"], utc=True)
        last = ts.max()
        age_hours = (ensure_utc(now_utc()) - last.to_pydatetime()).total_seconds() / 3600
        expected = pd.date_range(ts.min(), ts.max(), freq="h")
        missing_pct = (len(expected) - ts.nunique()) / max(1, len(expected)) * 100
        dup_rate = df.duplicated(subset=["city_id", "timestamp"]).mean() * 100
        return {
            "available": True,
            "rows": len(df),
            "last_data_timestamp": last.isoformat(),
            "data_age_hours": round(age_hours, 2),
            "is_stale": bool(age_hours > self.config.monitoring.max_data_age_hours),
            "missing_row_pct": round(float(missing_pct), 2),
            "duplicate_rate_pct": round(float(dup_rate), 3),
        }

    # -- model monitoring -------------------------------------------------
    def model_performance(self, city_id: str) -> dict[str, Any]:
        """Compare stored predictions against later-observed AQI."""
        preds = self.feature_store.read_features(PREDICTIONS_GROUP, city_id=city_id)
        actuals = self.feature_store.read_features(FEATURES_GROUP, city_id=city_id)
        if preds is None or preds.empty or actuals is None or actuals.empty:
            return {"available": False, "reason": "insufficient prediction/observation history"}
        p = preds[["timestamp", "predicted_aqi"]].copy()
        a = actuals[["timestamp", "aqi"]].copy()
        p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True)
        a["timestamp"] = pd.to_datetime(a["timestamp"], utc=True)
        merged = p.merge(a, on="timestamp", how="inner").dropna(subset=["predicted_aqi", "aqi"])
        window = self.config.monitoring.rolling_error_window_hours
        cutoff = ensure_utc(now_utc()) - pd.Timedelta(hours=window)
        recent = merged[merged["timestamp"] >= cutoff]
        use = recent if len(recent) >= 5 else merged
        if use.empty:
            return {"available": False, "reason": "no overlapping prediction/observation pairs"}
        m = regression_metrics(use["aqi"].to_numpy(), use["predicted_aqi"].to_numpy())
        by_cat = metrics_by_category(use["aqi"].to_numpy(), use["predicted_aqi"].to_numpy())
        return {
            "available": True,
            "n_pairs": len(use),
            "rolling_mae": m["mae"],
            "rolling_rmse": m["rmse"],
            "bias": m["bias"],
            "error_by_category": by_cat.to_dict(orient="records"),
        }

    # -- drift ------------------------------------------------------------
    def feature_drift(
        self, city_id: str, columns: list[str], split_fraction: float = 0.7
    ) -> dict[str, Any]:
        df = self.feature_store.read_features(FEATURES_GROUP, city_id=city_id)
        if df is None or len(df) < 50:
            return {"available": False, "reason": "insufficient rows for drift"}
        df = df.sort_values("timestamp")
        cut = int(len(df) * split_fraction)
        ref, cur = df.iloc[:cut], df.iloc[cut:]
        threshold = self.config.monitoring.drift_psi_threshold
        drift = {}
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                psi = population_stability_index(ref[col].to_numpy(), cur[col].to_numpy())
                drift[col] = {"psi": round(psi, 4), "drift": bool(psi > threshold)}
        return {"available": True, "threshold": threshold, "features": drift}

    # -- system status ----------------------------------------------------
    def system_status(self, city_id: str) -> dict[str, Any]:
        status: dict[str, Any] = {"generated_at": now_utc().isoformat(), "city_id": city_id}
        status["data"] = self.data_health(city_id)
        status["feature_store_backend"] = self.config.feature_store.backend
        status["model_registry_backend"] = self.config.model_registry.backend

        # Latest approved model.
        try:
            from pearls_aqi.registry import get_model_registry

            reg = get_model_registry(self.config)
            _, meta = reg.load_latest_approved(
                model_name=f"aqi_forecast_{city_id}", city_id=city_id
            )
            status["model"] = {
                "name": meta.model_name,
                "version": meta.version,
                "type": meta.model_type,
                "trained_at": meta.created_at,
                "status": meta.status,
            }
        except Exception as exc:
            status["model"] = {"available": False, "reason": str(exc)}

        # Last training report.
        try:
            report = read_json(f"artifacts/reports/{city_id}/training_report_latest.json")
            status["last_training_run"] = report.get("generated_at")
        except Exception:
            status["last_training_run"] = None
        return status

    def write_monitoring(self, city_id: str, record: dict[str, Any]) -> None:
        """Persist a flattened monitoring record to the monitoring group."""
        from pearls_aqi.storage.base import MONITORING_GROUP

        row = {
            "city_id": city_id,
            "timestamp": ensure_utc(now_utc()),
            "kind": record.get("kind", "status"),
            "payload": str(record),
        }
        try:
            self.feature_store.write_features(pd.DataFrame([row]), group=MONITORING_GROUP)
        except Exception as exc:
            logger.warning("Failed to persist monitoring record: %s", exc)


def latest_data_timestamp(df: pd.DataFrame) -> datetime | None:
    if df is None or df.empty or "timestamp" not in df.columns:
        return None
    return pd.to_datetime(df["timestamp"], utc=True).max().to_pydatetime()
