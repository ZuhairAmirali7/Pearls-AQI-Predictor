"""Best-effort in-process rate limiting.

A simple per-client token bucket for the heavier endpoints. This is intentionally
lightweight (single-process, in-memory) — production deployments should front the
API with a real rate limiter / API gateway.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from fastapi import HTTPException, Request

_RATE = 60  # tokens
_PER_SECONDS = 60.0  # per minute
_lock = threading.Lock()
_buckets: dict[str, tuple[float, float]] = defaultdict(lambda: (_RATE, time.monotonic()))


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "anonymous"


def rate_limiter(request: Request) -> None:
    """FastAPI dependency: raise 429 when a client exceeds the token budget."""
    key = _client_key(request)
    now = time.monotonic()
    with _lock:
        tokens, last = _buckets[key]
        tokens = min(_RATE, tokens + (now - last) * (_RATE / _PER_SECONDS))
        if tokens < 1.0:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
        _buckets[key] = (tokens - 1.0, now)
