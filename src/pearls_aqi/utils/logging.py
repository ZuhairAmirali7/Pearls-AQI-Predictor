"""Logging setup.

A single ``setup_logging`` call configures the ``pearls_aqi`` logger tree from
``config/logging.yaml`` (falling back to a sane console default). Modules obtain
loggers via ``get_logger(__name__)`` and never configure handlers themselves.
"""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Any

import yaml

_CONFIGURED = False
_DEFAULT_LOGGING_PATH = Path("config/logging.yaml")


def setup_logging(
    config_path: str | Path | None = None,
    level: str | None = None,
    force: bool = False,
) -> None:
    """Configure logging once for the whole process.

    Args:
        config_path: Path to a logging YAML config. Defaults to
            ``config/logging.yaml`` if it exists.
        level: Optional override for the ``pearls_aqi`` logger level (e.g. from
            the ``LOG_LEVEL`` environment variable).
        force: Reconfigure even if logging was already set up.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    path = Path(config_path) if config_path else _DEFAULT_LOGGING_PATH
    configured_from_file = False
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                cfg: dict[str, Any] = yaml.safe_load(fh)
            logging.config.dictConfig(cfg)
            configured_from_file = True
        except Exception:
            configured_from_file = False

    if not configured_from_file:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

    effective_level = level or os.getenv("LOG_LEVEL")
    if effective_level:
        logging.getLogger("pearls_aqi").setLevel(effective_level.upper())

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging has been configured."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
