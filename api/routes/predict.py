"""Ad-hoc prediction endpoint (testing / alternative inputs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_registry, resolve_location
from api.rate_limit import rate_limiter
from api.schemas import PredictRequest, PredictResponse
from pearls_aqi.exceptions import ModelNotFoundError, SchemaMismatchError

router = APIRouter(prefix="/api/v1", tags=["predict"])


@router.post("/predict", response_model=PredictResponse, dependencies=[Depends(rate_limiter)])
def predict(request: PredictRequest) -> PredictResponse:
    if not request.records:
        raise HTTPException(status_code=422, detail="`records` must be a non-empty list.")
    location = resolve_location(request.city)
    registry = get_registry()
    try:
        model, meta = registry.load_latest_approved(
            model_name=f"aqi_forecast_{location.city_id}", city_id=location.city_id
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    df = pd.DataFrame(request.records)
    names = meta.features or list(df.columns)
    overlap = sum(1 for c in names if c in df.columns) / len(names) if names else 0
    if overlap < 0.5:
        raise SchemaMismatchError(
            f"Only {overlap:.0%} of the model's features were supplied; cannot predict reliably."
        )
    for col in names:
        if col not in df.columns:
            df[col] = np.nan

    preds = np.asarray(model.predict(df[names])).tolist()
    return PredictResponse(
        model_name=meta.model_name,
        version=meta.version,
        horizon=model.horizon,
        predictions=preds,
    )
