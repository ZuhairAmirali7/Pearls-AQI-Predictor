"""Hopsworks model-registry backend.

Complete integration code, gated behind the optional ``hopsworks`` dependency
and a real account. Model *artifacts* are saved locally by the model's own
``save()`` then uploaded to the Hopsworks Model Registry with metadata + metrics.
Status transitions are tracked in the model description (Hopsworks has no native
status field), mirroring the local backend's lifecycle.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from pearls_aqi.config.models import AppConfig
from pearls_aqi.exceptions import ModelNotFoundError, ModelRegistryError
from pearls_aqi.registry.base import ModelMetadata, ModelStatus
from pearls_aqi.utils.io import write_json
from pearls_aqi.utils.logging import get_logger

logger = get_logger(__name__)


class HopsworksModelRegistry:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._project: Any = None
        self._mr: Any = None

    def _connect(self) -> None:
        if self._mr is not None:
            return
        try:
            import hopsworks
        except ImportError as exc:  # pragma: no cover
            raise ModelRegistryError(
                "hopsworks not installed. `pip install -e '.[hopsworks]'` or use local registry."
            ) from exc
        api_key = os.getenv("HOPSWORKS_API_KEY")
        if not api_key:
            raise ModelRegistryError("HOPSWORKS_API_KEY not set.")
        try:
            self._project = hopsworks.login(
                api_key_value=api_key, project=os.getenv("HOPSWORKS_PROJECT")
            )
            self._mr = self._project.get_model_registry()
        except Exception as exc:  # pragma: no cover
            raise ModelRegistryError(f"Hopsworks connection failed: {exc}") from exc

    def register(
        self,
        model: Any,
        metadata: ModelMetadata,
        model_card: str | None = None,
        status: ModelStatus = ModelStatus.CANDIDATE,
    ) -> ModelMetadata:
        self._connect()
        metadata.status = status.value
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model.save(tmp_path)
            write_json(metadata.to_dict(), tmp_path / "metadata.json")
            write_json(metadata.metrics, tmp_path / "metrics.json")
            if model_card:
                (tmp_path / "model_card.md").write_text(model_card, encoding="utf-8")
            try:
                mr_model = self._mr.python.create_model(
                    name=metadata.model_name,
                    metrics=_finite_metrics(metadata.metrics),
                    description=f"status={status.value}; type={metadata.model_type}",
                )
                mr_model.save(str(tmp_path))
                metadata.version = mr_model.version
            except Exception as exc:  # pragma: no cover
                raise ModelRegistryError(f"Hopsworks model save failed: {exc}") from exc
        logger.info(
            "Registered %s v%s to Hopsworks (%s).",
            metadata.model_name,
            metadata.version,
            status.value,
        )
        return metadata

    def set_status(self, model_name: str, version: int, status: ModelStatus) -> None:
        self._connect()
        try:
            model = self._mr.get_model(model_name, version=version)
            model.description = f"status={status.value}"
            model.update()
        except Exception as exc:  # pragma: no cover
            raise ModelNotFoundError(
                f"{model_name} v{version} not found in Hopsworks: {exc}"
            ) from exc

    def load(self, model_name: str, version: int | None = None) -> tuple[Any, ModelMetadata]:
        self._connect()
        model = (
            self._mr.get_model(model_name, version=version)
            if version
            else self._mr.get_best_model(model_name, "mae", "min")
        )
        return self._materialise(model)

    def load_latest_approved(
        self, model_name: str | None = None, city_id: str | None = None
    ) -> tuple[Any, ModelMetadata]:
        self._connect()
        if model_name is None:
            raise ModelRegistryError("Hopsworks backend requires an explicit model_name.")
        models = self._mr.get_models(model_name)
        approved = [m for m in models if "status=approved" in (m.description or "")]
        if not approved:
            raise ModelNotFoundError(f"No approved '{model_name}' model in Hopsworks.")
        best = max(approved, key=lambda m: m.version)
        return self._materialise(best)

    def list_versions(
        self, model_name: str | None = None
    ) -> list[ModelMetadata]:  # pragma: no cover
        self._connect()
        if model_name is None:
            return []
        out = []
        for m in self._mr.get_models(model_name):
            out.append(
                ModelMetadata(model_name=model_name, version=m.version, model_type="unknown")
            )
        return out

    def _materialise(self, mr_model: Any) -> tuple[Any, ModelMetadata]:  # pragma: no cover
        from pearls_aqi.models import load_model
        from pearls_aqi.utils.io import read_json

        local_dir = Path(mr_model.download())
        meta = ModelMetadata.from_dict(read_json(local_dir / "metadata.json"))
        model = load_model(meta.model_type, local_dir)
        return model, meta


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    import math

    return {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    }
