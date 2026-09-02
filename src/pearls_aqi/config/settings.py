"""Configuration loader.

Resolution order (later wins):
  1. ``config/config.yaml`` (or ``PEARLS_CONFIG_PATH``), falling back to
     ``config/config.example.yaml`` so the project runs out-of-the-box.
  2. Environment variables (``.env`` locally, GitHub Actions secrets in CI).

Only a small, documented set of environment variables override config — the
active location, provider/backend selection, and directories. Everything else
lives in YAML so behaviour is reproducible and reviewable.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from pearls_aqi.config.models import AppConfig
from pearls_aqi.exceptions import ConfigError

_DEFAULT_PATHS = ("config/config.yaml", "config/config.example.yaml")


def _find_config_path(explicit: str | Path | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise ConfigError(f"Config file not found: {p}")
        return p
    env_path = os.getenv("PEARLS_CONFIG_PATH")
    candidates = ([env_path] if env_path else []) + list(_DEFAULT_PATHS)
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise ConfigError(
        "No configuration file found. Expected one of: "
        + ", ".join(str(c) for c in candidates)
        + ". Copy config/config.example.yaml to config/config.yaml."
    )


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply the documented environment-variable overrides onto the raw dict."""
    raw.setdefault("location", {})
    loc = raw["location"]
    _set_if_env(loc, "city", "DEFAULT_CITY")
    _set_if_env(loc, "country", "DEFAULT_COUNTRY")
    _set_if_env(loc, "latitude", "DEFAULT_LATITUDE", cast=float)
    _set_if_env(loc, "longitude", "DEFAULT_LONGITUDE", cast=float)
    _set_if_env(loc, "timezone", "DEFAULT_TIMEZONE")

    raw.setdefault("providers", {})
    prov = raw["providers"]
    _set_if_env(prov, "air_quality", "AQI_PROVIDER")
    _set_if_env(prov, "weather", "WEATHER_PROVIDER")

    raw.setdefault("feature_store", {})
    fs = raw["feature_store"]
    _set_if_env(fs, "backend", "FEATURE_STORE_BACKEND")
    _set_if_env(fs, "local_dir", "LOCAL_FEATURE_STORE_DIR")

    raw.setdefault("model_registry", {})
    mr = raw["model_registry"]
    _set_if_env(mr, "backend", "MODEL_REGISTRY_BACKEND")
    _set_if_env(mr, "local_dir", "LOCAL_MODEL_REGISTRY_DIR")

    raw.setdefault("alerts", {})
    _set_if_env(raw["alerts"], "enabled", "ALERTS_ENABLED", cast=_as_bool)

    raw.setdefault("display", {})
    _set_if_env(raw["display"], "timezone", "DEFAULT_TIMEZONE")
    return raw


def _set_if_env(target: dict[str, Any], key: str, env_var: str, cast: Any = str) -> None:
    value = os.getenv(env_var)
    if value is not None and value != "":
        target[key] = cast(value)


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load, override, and validate the application configuration."""
    config_path = _find_config_path(path)
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse {config_path}: {exc}") from exc

    raw = _apply_env_overrides(raw)
    try:
        return AppConfig.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(f"Invalid configuration in {config_path}: {exc}") from exc


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return a cached, process-wide config (loaded on first access)."""
    return load_config()
