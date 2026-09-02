"""Model-registry interface, metadata schema, and status lifecycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ModelStatus(str, Enum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Everything needed to reproduce, audit, and safely load a model."""

    model_name: str
    version: int
    model_type: str
    status: str = ModelStatus.CANDIDATE.value
    created_at: str = ""
    training_start: str | None = None
    training_end: str | None = None
    feature_schema_version: str = "1.0.0"
    features: list[str] = field(default_factory=list)
    target: str = "aqi"
    forecast_horizon: int = 72
    city_id: str | None = None
    city: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    git_sha: str | None = None
    python_version: str | None = None
    dependencies: dict[str, str] = field(default_factory=dict)
    model_card_path: str | None = None
    artifact_size_bytes: int | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@runtime_checkable
class ModelRegistry(Protocol):
    """Persist, version, promote, and load models."""

    def register(
        self,
        model: Any,
        metadata: ModelMetadata,
        model_card: str | None = None,
        status: ModelStatus = ModelStatus.CANDIDATE,
    ) -> ModelMetadata:
        """Persist a trained model + metadata; return the stored metadata."""
        ...

    def load(self, model_name: str, version: int | None = None) -> tuple[Any, ModelMetadata]:
        """Load a specific model version (latest if ``version`` is None)."""
        ...

    def load_latest_approved(
        self, model_name: str | None = None, city_id: str | None = None
    ) -> tuple[Any, ModelMetadata]:
        """Load the newest APPROVED model (optionally filtered)."""
        ...

    def set_status(self, model_name: str, version: int, status: ModelStatus) -> None: ...

    def list_versions(self, model_name: str | None = None) -> list[ModelMetadata]: ...


def utc_now_iso() -> str:
    from pearls_aqi.utils.timeutils import now_utc

    return now_utc().isoformat()


def collect_environment() -> dict[str, str]:
    """Capture python + key dependency versions for reproducibility."""
    import platform
    from importlib.metadata import PackageNotFoundError, version

    deps: dict[str, str] = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scikit-learn", "tensorflow"):
        try:
            deps[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return deps


def git_sha() -> str | None:
    """Best-effort current git commit SHA (None outside a repo)."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        return sha or None
    except Exception:
        return None
