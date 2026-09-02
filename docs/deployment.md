# Deployment Guide

The system is modular: dashboard, API, storage, and orchestration deploy
independently. Everything runs locally with **no paid services**; the cloud
options below are optional upgrades.

## Environment variables

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `APP_ENV` | development | Environment label. |
| `LOG_LEVEL` | INFO | Logging level. |
| `AQI_PROVIDER` / `WEATHER_PROVIDER` | openmeteo | `openmeteo` \| `openweather` \| `sample`. |
| `OPENWEATHER_API_KEY` | — | Required only for `openweather`. |
| `FEATURE_STORE_BACKEND` | local | `local` \| `hopsworks`. |
| `MODEL_REGISTRY_BACKEND` | local | `local` \| `hopsworks`. |
| `HOPSWORKS_API_KEY` / `HOPSWORKS_PROJECT` | — | Required only for `hopsworks`. |
| `PEARLS_API_BASE_URL` | — | Dashboard → API base URL (empty = direct library mode). |
| `ALERTS_ENABLED` | false (env) | Enable alert dispatch. |
| `ALERT_EMAIL_TO`, `SMTP_*`, `ALERT_WEBHOOK_URL` | — | Optional alert channels. |

Local: copy `.env.example` → `.env`. CI/cloud: set as secrets (never commit).

## 1. Local (Docker Compose)

```bash
cp config/config.example.yaml config/config.yaml
cp .env.example .env
docker compose up --build
# API:       http://localhost:8000  (/docs for OpenAPI)
# Dashboard: http://localhost:8501
```

One image serves both services (command selects which); they share a data
volume. A non-root user runs the app and container health checks are configured.

Before first forecast, seed + train inside the running API container (or locally):

```bash
docker compose exec api python scripts/seed_sample_data.py --city Karachi
docker compose exec api python scripts/run_training_pipeline.py --city Karachi --no-tune
```

## 2. Dashboard → Streamlit Community Cloud

1. Push the repo to GitHub.
2. On <https://share.streamlit.io>, create an app pointing at
   `app/streamlit_app.py`.
3. Add secrets (Settings → Secrets): `PEARLS_API_BASE_URL` (your deployed API),
   plus provider/Hopsworks keys if used.
4. Deploys automatically on push. No key needed for sample/Open-Meteo mode.

## 3. API → Render / Railway / Cloud Run

Container from the provided `Dockerfile` (start command
`uvicorn api.main:app --host 0.0.0.0 --port $PORT`):

- **Render:** New → Web Service → Docker → set env vars → set a Deploy Hook and
  put it in the `RENDER_DEPLOY_HOOK` GitHub secret for `deploy.yml`.
- **Railway:** New Project → Deploy from repo (Docker) → set variables.
- **Cloud Run:** `gcloud run deploy pearls-api --source . --port 8000
  --allow-unauthenticated` and set env vars.

Set `FEATURE_STORE_BACKEND=hopsworks` (+ keys) so the API reads persisted
features/models; otherwise seed/train on the instance.

## 4. Storage → Hopsworks

1. Create a free project at <https://app.hopsworks.ai> and an API key.
2. `pip install -e ".[hopsworks]"`.
3. Set `FEATURE_STORE_BACKEND=hopsworks`, `MODEL_REGISTRY_BACKEND=hopsworks`,
   `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`.
4. Feature groups (`raw_air_quality`, `raw_weather`, `aqi_features`, …) are
   created on first write with PK `city_id+timestamp`, event-time `timestamp`.
   The trained model is uploaded to the Hopsworks Model Registry.

If Hopsworks is unavailable at runtime, the factories log a warning and fall back
to the local backends.

## 5. Orchestration → GitHub Actions

Workflows in `.github/workflows/`:

| Workflow | Schedule | Does |
| -------- | -------- | ---- |
| `ci.yml` | push/PR | Ruff, Black, mypy, pytest, Docker build. |
| `feature_pipeline.yml` | hourly `0 * * * *` | Ingest + build features. |
| `training_pipeline.yml` | daily `30 1 * * *` | Train, evaluate, conditional promote. |
| `deploy.yml` | push to `main` | Test gate → build/push image → trigger deploys. |

**Repository secrets** (Settings → Secrets and variables → Actions):
`OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`,
`RENDER_DEPLOY_HOOK` (optional). **Variables**: `FEATURE_STORE_BACKEND`,
`MODEL_REGISTRY_BACKEND`, `AQI_PROVIDER`, `WEATHER_PROVIDER`. Without secrets the
pipelines still run (Open-Meteo is keyless; local store is ephemeral in CI).

## Rollback

The registry keeps every version with a status. To roll back, set the current
`approved` model to `archived` and re-approve a previous version, or pin
`--model-version` in `generate_forecast.py` / the `/api/v1/forecast` request.
