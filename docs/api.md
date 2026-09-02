# API Reference

FastAPI backend. Interactive OpenAPI docs at **`/docs`** (Swagger) and
**`/redoc`** when the server is running. Base URL below assumes
`http://localhost:8000`.

```bash
make api   # or: uvicorn api.main:app --reload
```

All `/api/v1/*` endpoints accept an optional `city` query parameter (defaults to
the configured city). Timestamps are UTC ISO-8601. Errors return a JSON body
`{"error": "...", "type": "..."}` with an appropriate status (404 no model/data,
422 validation/schema mismatch, 502 provider error, 503 not ready).

## Health

### `GET /health`
Liveness. `200 {"status": "ok", "version": "..."}`.

### `GET /ready`
Readiness — checks config loaded, feature store reachable, a model is available,
and the feature schema is compatible. `200` when all pass, else `503` with the
per-check booleans.

```json
{ "ready": true, "checks": { "config": true, "feature_store": true,
  "model_available": true, "schema_compatible": true } }
```

## `GET /api/v1/current?city=Karachi`
Latest observed AQI, category, pollutant concentrations, and weather.

```json
{ "city": "Karachi", "timestamp": "2026-07-24T12:00:00Z", "aqi": 118.0,
  "category": "Unhealthy for Sensitive Groups", "dominant_pollutant": "pm2_5",
  "pollutants": { "pm2_5": 42.1, "pm10": 88.0, "o3": 61.0, "no2": 19.0, "so2": 8.0, "co": 540.0 },
  "weather": { "temperature": 33.2, "humidity": 58, "wind_speed": 3.1 } }
```

## `GET /api/v1/forecast?city=Karachi&horizon=72&model_version=`
72-hour (or `horizon`) AQI forecast from the approved model, plus hazard alerts.

```json
{
  "city": "Karachi",
  "generated_at": "2026-07-24T12:00:00Z",
  "model_name": "aqi_forecast_karachi_pakistan",
  "model_version": 1,
  "forecast": [
    { "timestamp": "2026-07-24T13:00:00Z", "horizon_hour": 1,
      "predicted_aqi": 121.5, "lower_bound": 104.2, "upper_bound": 138.8,
      "category": "Unhealthy for Sensitive Groups", "dominant_pollutant": "pm2_5" }
  ],
  "alerts": [ { "type": "threshold_exceeded", "severity": "unhealthy",
                "message": "AQI is forecast to reach 176 ...", "peak_aqi": 176.0 } ]
}
```

```bash
curl "http://localhost:8000/api/v1/forecast?city=Karachi&horizon=72"
```

## `GET /api/v1/history?city=Karachi&start=2026-01-01&end=2026-02-01&frequency=hourly`
Historical AQI/pollutant/weather rows from the feature store. `frequency` may be
`hourly` (default) or `daily` (daily means).

## `GET /api/v1/model?city=Karachi`
Approved model metadata (name, version, type, status, training range, feature
count, git SHA, dependencies).

## `GET /api/v1/model/metrics?city=Karachi`
Evaluation metrics for the approved model (validation/test MAE, RMSE, R²,
hazard MAE, per-horizon MAE).

## `GET /api/v1/explanations?city=Karachi&horizon_index=0`
Global feature importance, a local explanation for the current forecast point,
and a plain-language summary. Method: permutation/occlusion (SHAP when
installed). *Importances reflect model behaviour, not causation.*

## `POST /api/v1/predict`
Run the approved model on caller-supplied features (testing / alternative inputs).
Rate-limited. Request:

```json
{ "city": "Karachi", "features": { "aqi": 120, "pm2_5": 45, "hour": 14, "...": 0 } }
```

Response: the 72-hour prediction array. Returns `422` on feature-schema mismatch
(< 80% of the model's features supplied).

## CORS & versioning
CORS is dev-permissive (`*`) — restrict `allow_origins` in production. The API is
versioned under `/api/v1`; `/health` and `/ready` are unversioned.
