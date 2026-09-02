"""FastAPI application entrypoint.

    uvicorn api.main:app --reload

Serves the Pearls AQI Predictor REST API (OpenAPI docs at ``/docs``). All
business logic lives in ``pearls_aqi``; routes are thin adapters.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import (
    current,
    explanations,
    forecast,
    health,
    history,
    model,
    predict,
)
from pearls_aqi import __version__
from pearls_aqi.exceptions import (
    ModelNotFoundError,
    PearlsError,
    ProviderError,
    SchemaMismatchError,
    ValidationError,
)
from pearls_aqi.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("pearls_aqi.api")

app = FastAPI(
    title="Pearls AQI Predictor API",
    description=(
        "Serverless-style API serving 72-hour US EPA AQI forecasts, current "
        "conditions, history, model metadata, explanations, and hazard alerts.\n\n"
        "**Disclaimer:** forecasts are model estimates, not a substitute for "
        "official air-quality warnings or medical advice."
    ),
    version=__version__,
    contact={"name": "Pearls AQI Predictor"},
    license_info={"name": "MIT"},
)

# CORS — permissive by default (dashboard is a separate origin); tighten in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Exception handlers: map domain errors to clean JSON ---
@app.exception_handler(ModelNotFoundError)
async def _model_not_found(_: Request, exc: ModelNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": "model_unavailable", "detail": str(exc)})


@app.exception_handler(SchemaMismatchError)
async def _schema_mismatch(_: Request, exc: SchemaMismatchError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "schema_mismatch", "detail": str(exc)})


@app.exception_handler(ValidationError)
async def _validation(_: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "validation_error", "detail": str(exc)})


@app.exception_handler(ProviderError)
async def _provider(_: Request, exc: ProviderError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"error": "provider_error", "detail": str(exc)})


@app.exception_handler(PearlsError)
async def _pearls(_: Request, exc: PearlsError) -> JSONResponse:
    logger.error("Unhandled PearlsError: %s", exc)
    return JSONResponse(status_code=400, content={"error": "application_error", "detail": str(exc)})


# --- Routers ---
app.include_router(health.router)
app.include_router(current.router)
app.include_router(forecast.router)
app.include_router(history.router)
app.include_router(model.router)
app.include_router(explanations.router)
app.include_router(predict.router)


@app.get("/", tags=["health"])
def root() -> dict[str, str]:
    return {
        "service": "pearls-aqi-predictor",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }
