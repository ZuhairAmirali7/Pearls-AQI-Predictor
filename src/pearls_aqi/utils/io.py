"""Small filesystem helpers for Parquet / JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed; return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to Parquet, creating parent directories."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False)
    return p


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Parquet file, returning an empty frame if it does not exist."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def write_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Serialise an object to JSON on disk (dates handled via ``default=str``)."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=str, sort_keys=False)
    return p


def read_json(path: str | Path) -> Any:
    """Read a JSON file from disk."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)
