"""Local filesystem model registry (default backend).

Layout::

    artifacts/models/
      registry.json                      # index: all versions + status
      <model_name>/
        version_001/
          model.joblib | model_tf/       # written by the model's own save()
          preprocessing.joblib           # (optional, model-managed)
          metadata.json
          metrics.json
          model_card.md

``load_latest_approved`` consults the index for APPROVED status — never just the
newest file on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pearls_aqi.exceptions import ModelNotFoundError, ModelRegistryError
from pearls_aqi.registry.base import ModelMetadata, ModelStatus
from pearls_aqi.utils.io import ensure_dir, read_json, write_json
from pearls_aqi.utils.logging import get_logger

logger = get_logger(__name__)


class LocalModelRegistry:
    def __init__(self, local_dir: str | Path = "artifacts/models") -> None:
        self.root = ensure_dir(local_dir)
        self.index_path = self.root / "registry.json"

    # -- index ------------------------------------------------------------
    def _read_index(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        try:
            return read_json(self.index_path)
        except Exception as exc:
            raise ModelRegistryError(f"Corrupt registry index {self.index_path}: {exc}") from exc

    def _write_index(self, entries: list[dict[str, Any]]) -> None:
        write_json(entries, self.index_path)

    def _next_version(self, model_name: str) -> int:
        versions = [e["version"] for e in self._read_index() if e["model_name"] == model_name]
        return (max(versions) + 1) if versions else 1

    def _version_dir(self, model_name: str, version: int) -> Path:
        return self.root / model_name / f"version_{version:03d}"

    # -- write ------------------------------------------------------------
    def register(
        self,
        model: Any,
        metadata: ModelMetadata,
        model_card: str | None = None,
        status: ModelStatus = ModelStatus.CANDIDATE,
    ) -> ModelMetadata:
        version = metadata.version or self._next_version(metadata.model_name)
        metadata.version = version
        metadata.status = status.value
        vdir = ensure_dir(self._version_dir(metadata.model_name, version))

        # Delegate artifact persistence to the model (handles sklearn vs TF).
        try:
            model.save(vdir)
        except AttributeError as exc:
            raise ModelRegistryError(
                f"Model of type {type(model).__name__} has no .save(dir) method."
            ) from exc

        metadata.artifact_size_bytes = _dir_size(vdir)
        if model_card:
            card_path = vdir / "model_card.md"
            card_path.write_text(model_card, encoding="utf-8")
            metadata.model_card_path = str(card_path)

        write_json(metadata.to_dict(), vdir / "metadata.json")
        write_json(metadata.metrics, vdir / "metrics.json")

        index = self._read_index()
        index = [
            e
            for e in index
            if not (e["model_name"] == metadata.model_name and e["version"] == version)
        ]
        index.append(metadata.to_dict())
        self._write_index(index)
        logger.info(
            "Registered %s v%d as %s (%d bytes).",
            metadata.model_name,
            version,
            status.value,
            metadata.artifact_size_bytes or 0,
        )
        return metadata

    def set_status(self, model_name: str, version: int, status: ModelStatus) -> None:
        index = self._read_index()
        found = False
        for entry in index:
            if entry["model_name"] == model_name and entry["version"] == version:
                entry["status"] = status.value
                found = True
        if not found:
            raise ModelNotFoundError(f"{model_name} v{version} not in registry.")
        self._write_index(index)
        # also update the version's metadata.json
        meta_path = self._version_dir(model_name, version) / "metadata.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            meta["status"] = status.value
            write_json(meta, meta_path)
        logger.info("Set %s v%d status -> %s", model_name, version, status.value)

    # -- read -------------------------------------------------------------
    def load(self, model_name: str, version: int | None = None) -> tuple[Any, ModelMetadata]:
        entries = [e for e in self._read_index() if e["model_name"] == model_name]
        if not entries:
            raise ModelNotFoundError(f"No versions registered for model '{model_name}'.")
        if version is None:
            entry = max(entries, key=lambda e: e["version"])
        else:
            match = [e for e in entries if e["version"] == version]
            if not match:
                raise ModelNotFoundError(f"{model_name} v{version} not found.")
            entry = match[0]
        return self._load_entry(entry)

    def load_latest_approved(
        self, model_name: str | None = None, city_id: str | None = None
    ) -> tuple[Any, ModelMetadata]:
        entries = [
            e
            for e in self._read_index()
            if e["status"] == ModelStatus.APPROVED.value
            and (model_name is None or e["model_name"] == model_name)
            and (city_id is None or e.get("city_id") == city_id)
        ]
        if not entries:
            raise ModelNotFoundError(
                "No APPROVED model found"
                + (f" for city_id={city_id}" if city_id else "")
                + ". Run the training pipeline first."
            )
        entry = max(entries, key=lambda e: (e.get("created_at", ""), e["version"]))
        return self._load_entry(entry)

    def list_versions(self, model_name: str | None = None) -> list[ModelMetadata]:
        return [
            ModelMetadata.from_dict(e)
            for e in self._read_index()
            if model_name is None or e["model_name"] == model_name
        ]

    def _load_entry(self, entry: dict[str, Any]) -> tuple[Any, ModelMetadata]:
        metadata = ModelMetadata.from_dict(entry)
        vdir = self._version_dir(metadata.model_name, metadata.version)
        if not vdir.exists():
            raise ModelNotFoundError(f"Artifact directory missing: {vdir}")
        # Lazy import to avoid registry -> models circular dependency.
        from pearls_aqi.models import load_model

        model = load_model(metadata.model_type, vdir)
        return model, metadata


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
