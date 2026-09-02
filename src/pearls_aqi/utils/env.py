"""Minimal ``.env`` loader (dependency-free).

Loads ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ`` without
overriding variables already set in the real environment (so CI secrets win).
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env", override: bool = False) -> int:
    """Load a .env file into os.environ. Returns the number of keys set."""
    p = Path(path)
    if not p.exists():
        return 0
    count = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            count += 1
    return count
