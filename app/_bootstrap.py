"""Ensure the repo root is importable so ``import app.*`` works under both
``streamlit run app/streamlit_app.py`` and ``pytest`` / plain Python.

Every page imports this first.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root() -> Path:
    root = Path(__file__).resolve()
    while root.parent != root and not (root / "pyproject.toml").exists():
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


ROOT = ensure_repo_root()
