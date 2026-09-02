# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Pearls AQI Predictor — single image serving both the FastAPI backend and the
# Streamlit dashboard (the running command selects which). Core dependencies
# only; TensorFlow/SHAP/Hopsworks are optional extras and intentionally omitted
# to keep the image small. Runs as a non-root user.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    PEARLS_CONFIG_PATH=config/config.yaml

WORKDIR /app

# System deps: curl for container health checks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# --- Dependency layer (cached unless packaging metadata changes) ---
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip && pip install .

# --- Application code ---
COPY config/ ./config/
COPY api/ ./api/
COPY app/ ./app/
COPY scripts/ ./scripts/

# Config: ship the example as the default config if none is mounted.
RUN cp -n config/config.example.yaml config/config.yaml || true

# Writable dirs for local feature store / artifacts, owned by the app user.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/local /app/artifacts \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

# Default: FastAPI backend. docker-compose overrides the command for Streamlit.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
