"""Training pipeline: features → targets → chronological split → train/compare →
select → register → (conditionally) promote.

The test set is used for final reporting only — model selection is on the
validation split. Each model is trained inside a try/except so one failure never
aborts the comparison (graceful degradation, e.g. TensorFlow not installed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from pearls_aqi.config.models import AppConfig
from pearls_aqi.data.domain import Location
from pearls_aqi.evaluation.metrics import (
    hazard_metrics,
    metrics_by_horizon,
    regression_metrics,
)
from pearls_aqi.evaluation.selection import evaluate_promotion, select_best_model
from pearls_aqi.evaluation.splits import chronological_split
from pearls_aqi.exceptions import TrainingError
from pearls_aqi.features.engineering import select_feature_columns
from pearls_aqi.features.targets import build_direct_targets, target_columns, trainable_rows
from pearls_aqi.models import build_model, is_available
from pearls_aqi.registry.base import (
    ModelMetadata,
    ModelStatus,
    collect_environment,
    git_sha,
    utc_now_iso,
)
from pearls_aqi.utils.io import write_json
from pearls_aqi.utils.logging import get_logger
from pearls_aqi.utils.timeutils import now_utc

logger = get_logger(__name__)

_TUNED = {"ridge", "elastic_net", "random_forest"}  # models that run a search


@dataclass
class ModelResult:
    name: str
    model_type: str
    validation: dict[str, float] = field(default_factory=dict)
    test: dict[str, float] = field(default_factory=dict)
    hazard: dict[str, float] = field(default_factory=dict)
    by_horizon_mae: list[float] = field(default_factory=list)
    best_params: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "validation": self.validation,
            "test": self.test,
            "hazard": self.hazard,
            "best_params": self.best_params,
            "error": self.error,
        }


@dataclass
class TrainingReport:
    city: str
    model_name: str
    selected: str | None
    results: list[ModelResult]
    promotion: dict[str, Any]
    split_summary: dict[str, int]
    horizon: int
    generated_at: datetime
    registered_version: int | None = None
    registered_status: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "city": self.city,
            "model_name": self.model_name,
            "selected": self.selected,
            "horizon": self.horizon,
            "registered_version": self.registered_version,
            "registered_status": self.registered_status,
            "promotion": self.promotion,
            "split_summary": self.split_summary,
            "generated_at": self.generated_at.isoformat(),
            "results": [r.as_dict() for r in self.results],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Training Report — {self.city}",
            "",
            f"- Generated: {self.generated_at.isoformat()}",
            f"- Registered model: `{self.model_name}` "
            f"v{self.registered_version} ({self.registered_status})",
            f"- Selected model type: **{self.selected}**",
            f"- Horizon: {self.horizon} h",
            f"- Split: {self.split_summary}",
            "",
            "## Model comparison (validation)",
            "",
            "| Model | Val MAE | Val RMSE | Val R² | Test MAE | Hazard MAE |",
            "| ----- | ------: | -------: | -----: | -------: | ---------: |",
        ]
        for r in sorted(self.results, key=lambda x: x.validation.get("mae", float("inf"))):
            if r.error:
                lines.append(f"| {r.name} | — | — | — | — | _{r.error[:40]}_ |")
                continue
            lines.append(
                f"| {r.name} | {r.validation.get('mae', float('nan')):.2f} | "
                f"{r.validation.get('rmse', float('nan')):.2f} | "
                f"{r.validation.get('r2', float('nan')):.3f} | "
                f"{r.test.get('mae', float('nan')):.2f} | "
                f"{r.hazard.get('hazard_mae', float('nan')):.2f} |"
            )
        lines += ["", f"**Promotion decision:** {self.promotion}"]
        return "\n".join(lines)


class TrainingPipeline:
    def __init__(self, config: AppConfig, feature_store: Any = None, registry: Any = None) -> None:
        self.config = config
        self._fs = feature_store
        self._registry = registry

    @property
    def feature_store(self):
        if self._fs is None:
            from pearls_aqi.storage import get_feature_store

            self._fs = get_feature_store(self.config)
        return self._fs

    @property
    def registry(self):
        if self._registry is None:
            from pearls_aqi.registry import get_model_registry

            self._registry = get_model_registry(self.config)
        return self._registry

    def run(
        self,
        location: Location,
        horizon: int | None = None,
        models: list[str] | None = None,
        tune: bool | None = None,
        max_rows: int | None = None,
        report_dir: str = "artifacts/reports",
    ) -> TrainingReport:
        cfg = self.config
        horizon = horizon or cfg.forecast.horizon_hours
        model_names = models or cfg.training.models
        tune = cfg.training.random_search_iterations > 0 if tune is None else tune
        city_id = location.city_id
        model_name = f"aqi_forecast_{city_id}"

        # 1. Load features.
        df = self.feature_store.get_training_data(city_id=city_id)
        if df is None or len(df) < cfg.training.min_training_rows:
            raise TrainingError(
                f"Insufficient training data for {location.city}: "
                f"{0 if df is None else len(df)} rows (< {cfg.training.min_training_rows}). "
                "Run a historical backfill first."
            )
        if max_rows:
            df = df.tail(max_rows).reset_index(drop=True)

        # 2. Targets + trainable rows.
        tcols = target_columns(horizon)
        df = build_direct_targets(df, horizon)
        df = trainable_rows(df, horizon)
        if len(df) < cfg.training.min_training_rows:
            raise TrainingError(
                f"After target construction only {len(df)} trainable rows remain "
                f"(< {cfg.training.min_training_rows})."
            )
        feature_cols = select_feature_columns(df, tcols)

        # 3. Chronological split.
        split = chronological_split(
            df,
            cfg.training.train_fraction,
            cfg.training.validation_fraction,
            cfg.training.test_fraction,
            gap_hours=cfg.training.split_gap_hours,
        )
        Xtr, ytr = split.train[feature_cols], split.train[tcols]
        Xval, yval = split.validation[feature_cols], split.validation[tcols]
        Xte, yte = split.test[feature_cols], split.test[tcols]
        logger.info("Training split: %s | features=%d", split.summary(), len(feature_cols))

        # 4. Train + evaluate each model.
        results: list[ModelResult] = []
        fitted: dict[str, Any] = {}
        for name in model_names:
            if not is_available(name):
                logger.warning("Model '%s' unavailable in this environment; skipping.", name)
                results.append(ModelResult(name=name, model_type=name, error="unavailable"))
                continue
            res, model = self._train_one(name, horizon, tune, Xtr, ytr, Xval, yval, Xte, yte)
            results.append(res)
            if model is not None:
                fitted[name] = model

        # 5. Select best by validation MAE.
        selectable = [
            {"name": r.name, "validation": r.validation, "hazard": r.hazard, "result": r}
            for r in results
            if not r.error and np.isfinite(r.validation.get("mae", np.nan))
        ]
        if not selectable:
            raise TrainingError("No model produced a valid validation score.")
        best = select_best_model(selectable)
        best_name = best["name"]
        logger.info("Selected best model: %s (%s)", best_name, best["selection_rationale"])

        # 6. Refit winner on train+val for the deployed artifact; report on test.
        deployed = build_model(best_name, horizon, cfg.training, tune=False)
        X_fit = pd.concat([Xtr, Xval])
        y_fit = pd.concat([ytr, yval])
        deployed.fit(X_fit, y_fit)
        smoke_ok = self._smoke_test(deployed, Xte if len(Xte) else Xtr)

        best_result = next(r for r in results if r.name == best_name)
        metadata = self._build_metadata(
            model_name, best_name, deployed, feature_cols, horizon, location, split, best_result, df
        )
        model_card = self._render_model_card(metadata, results, best_result)

        # 7. Register as candidate.
        stored = self.registry.register(
            deployed, metadata, model_card=model_card, status=ModelStatus.CANDIDATE
        )

        # 8. Promotion decision vs incumbent approved model.
        incumbent_metrics = self._incumbent_metrics(model_name, city_id)
        decision = evaluate_promotion(
            candidate=self._as_promotion_dict(best_result),
            incumbent=incumbent_metrics,
            minimum_mae_improvement_percent=cfg.model_promotion.minimum_mae_improvement_percent,
            maximum_hazard_mae_degradation_percent=cfg.model_promotion.maximum_hazard_mae_degradation_percent,
            smoke_ok=smoke_ok,
            require_smoke_test=cfg.model_promotion.require_smoke_test,
            artifacts_complete=True,
        )
        status = ModelStatus.CANDIDATE
        if decision.promote:
            self._archive_previous_approved(model_name, city_id)
            self.registry.set_status(model_name, stored.version, ModelStatus.APPROVED)
            status = ModelStatus.APPROVED
        logger.info("Promotion: %s — %s", decision.promote, "; ".join(decision.reasons))

        report = TrainingReport(
            city=location.city,
            model_name=model_name,
            selected=best_name,
            results=results,
            promotion=decision.as_dict(),
            split_summary=split.summary(),
            horizon=horizon,
            generated_at=now_utc(),
            registered_version=stored.version,
            registered_status=status.value,
        )
        self._write_report(report, report_dir, city_id)
        return report

    # -- internals --------------------------------------------------------
    def _train_one(self, name, horizon, tune, Xtr, ytr, Xval, yval, Xte, yte):
        try:
            model = build_model(name, horizon, self.config.training, tune=(tune and name in _TUNED))
            model.fit(Xtr, ytr)
            val_pred = model.predict(Xval) if len(Xval) else np.empty((0, horizon))
            test_pred = model.predict(Xte) if len(Xte) else np.empty((0, horizon))
            res = ModelResult(
                name=name,
                model_type=model.model_type,
                validation=regression_metrics(yval.to_numpy(), val_pred) if len(Xval) else {},
                test=regression_metrics(yte.to_numpy(), test_pred) if len(Xte) else {},
                hazard=(
                    hazard_metrics(
                        yval.to_numpy(), val_pred, threshold=self.config.alerts.unhealthy_threshold
                    )
                    if len(Xval)
                    else {}
                ),
                by_horizon_mae=(
                    metrics_by_horizon(yval.to_numpy(), val_pred)["mae"].tolist()
                    if len(Xval)
                    else []
                ),
                best_params=getattr(model, "best_params_", {}) or {},
            )
            logger.info(
                "Trained %-22s val_mae=%.3f test_mae=%.3f",
                name,
                res.validation.get("mae", float("nan")),
                res.test.get("mae", float("nan")),
            )
            return res, model
        except Exception as exc:
            logger.exception("Model '%s' failed to train: %s", name, exc)
            return ModelResult(name=name, model_type=name, error=str(exc)), None

    def _smoke_test(self, model, X: pd.DataFrame) -> bool:
        try:
            if len(X) == 0:
                return True
            preds = model.predict(X.head(1))
            return bool(np.all(np.isfinite(preds)) and preds.shape[1] == model.horizon)
        except Exception as exc:
            logger.error("Smoke test failed: %s", exc)
            return False

    def _build_metadata(
        self, model_name, best_name, model, feature_cols, horizon, location, split, best_result, df
    ) -> ModelMetadata:
        metrics = {
            "validation_mae": best_result.validation.get("mae"),
            "validation_rmse": best_result.validation.get("rmse"),
            "validation_r2": best_result.validation.get("r2"),
            "test_mae": best_result.test.get("mae"),
            "test_rmse": best_result.test.get("rmse"),
            "test_r2": best_result.test.get("r2"),
            "mae": best_result.validation.get("mae"),  # used by selection/promotion
            "rmse": best_result.validation.get("rmse"),
            "r2": best_result.validation.get("r2"),
            "hazard_mae": best_result.hazard.get("hazard_mae"),
            "by_horizon_mae": best_result.by_horizon_mae,
        }
        ts = pd.to_datetime(df["timestamp"], utc=True)
        return ModelMetadata(
            model_name=model_name,
            version=0,  # assigned by registry
            model_type=best_name,
            created_at=utc_now_iso(),
            training_start=ts.min().isoformat() if len(ts) else None,
            training_end=ts.max().isoformat() if len(ts) else None,
            feature_schema_version=self.config.project.feature_pipeline_version,
            features=feature_cols,
            target="aqi",
            forecast_horizon=horizon,
            city_id=location.city_id,
            city=location.city,
            metrics={k: v for k, v in metrics.items() if v is not None},
            git_sha=git_sha(),
            python_version=collect_environment().get("python"),
            dependencies=collect_environment(),
        )

    def _as_promotion_dict(self, r: ModelResult) -> dict[str, Any]:
        return {
            "validation": {"mae": r.validation.get("mae")},
            "hazard": {"hazard_mae": r.hazard.get("hazard_mae")},
        }

    def _incumbent_metrics(self, model_name: str, city_id: str) -> dict[str, Any] | None:
        try:
            _, meta = self.registry.load_latest_approved(model_name=model_name, city_id=city_id)
        except Exception:
            return None
        return {
            "validation": {"mae": meta.metrics.get("validation_mae") or meta.metrics.get("mae")},
            "hazard": {"hazard_mae": meta.metrics.get("hazard_mae")},
        }

    def _archive_previous_approved(self, model_name: str, city_id: str) -> None:
        try:
            _, meta = self.registry.load_latest_approved(model_name=model_name, city_id=city_id)
            self.registry.set_status(model_name, meta.version, ModelStatus.ARCHIVED)
        except Exception:
            return

    def _render_model_card(self, meta: ModelMetadata, results, best_result) -> str:
        m = meta.metrics
        lines = [
            f"# Model Card — {meta.model_name}",
            "",
            f"- **Model type:** {meta.model_type}",
            f"- **Version:** {meta.version}",
            f"- **City:** {meta.city} (`{meta.city_id}`)",
            f"- **Forecast horizon:** {meta.forecast_horizon} hours (direct multi-horizon)",
            f"- **Trained:** {meta.created_at}",
            f"- **Training range:** {meta.training_start} → {meta.training_end}",
            f"- **# features:** {len(meta.features)}",
            f"- **Git SHA:** {meta.git_sha or 'n/a'}",
            "",
            "## Metrics (validation / test)",
            f"- MAE: {m.get('validation_mae', float('nan')):.2f} / {m.get('test_mae', float('nan')):.2f}",
            f"- RMSE: {m.get('validation_rmse', float('nan')):.2f} / {m.get('test_rmse', float('nan')):.2f}",
            f"- R²: {m.get('validation_r2', float('nan')):.3f} / {m.get('test_r2', float('nan')):.3f}",
            f"- Hazard-period MAE: {m.get('hazard_mae', float('nan')):.2f}",
            "",
            "## Intended use & limitations",
            "Short-term (72h) AQI guidance for the configured city. **Not** a "
            "substitute for official air-quality warnings or medical advice. "
            "Skill degrades with horizon; extreme episodes are under-represented.",
            "",
            "## Explainability",
            "Feature attribution via SHAP (permutation-importance fallback). "
            "Importances reflect model behaviour, not proven causation.",
        ]
        return "\n".join(lines)

    def _write_report(self, report: TrainingReport, report_dir: str, city_id: str) -> None:
        from pathlib import Path

        from pearls_aqi.utils.io import ensure_dir

        stamp = report.generated_at.strftime("%Y%m%dT%H%M%SZ")
        base = ensure_dir(f"{report_dir}/{city_id}")
        write_json(report.as_dict(), base / f"training_report_{stamp}.json")
        Path(base / f"training_report_{stamp}.md").write_text(
            report.to_markdown(), encoding="utf-8"
        )
        # Also write a stable "latest" copy for the dashboard.
        write_json(report.as_dict(), base / "training_report_latest.json")
