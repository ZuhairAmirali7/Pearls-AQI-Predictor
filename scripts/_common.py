"""Shared setup for CLI scripts: load .env, configure logging, resolve config.

Every script calls :func:`bootstrap` first so behaviour is consistent (UTC logs,
env-driven config, clear errors).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from pearls_aqi.config import AppConfig, load_config
from pearls_aqi.data.domain import Location
from pearls_aqi.utils.env import load_dotenv
from pearls_aqi.utils.logging import get_logger, setup_logging


def bootstrap() -> AppConfig:
    """Load .env, set up logging, and return the validated application config."""
    load_dotenv()
    setup_logging()
    return load_config()


def resolve_location(config: AppConfig, city: str | None) -> Location:
    """Return the Location for ``city`` (or the active location if None)."""
    if city is None or city == config.location.city:
        return config.active_location
    return config.location_for_city(city)


def parse_date(value: str) -> datetime:
    """Parse a YYYY-MM-DD (or ISO) date string as a UTC datetime."""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date {value!r}; use YYYY-MM-DD.") from exc
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


logger = get_logger("pearls_aqi.scripts")
