# Pearls AQI Predictor — Final Report

> **Health disclaimer.** All forecasts are model estimates for informational
> purposes only and **do not replace official air-quality warnings or medical
> advice.**

## Executive summary

Pearls AQI Predictor is an end-to-end, primarily-serverless system that forecasts
the US EPA Air Quality Index for a configurable city for the next 72 hours. It
collects weather and pollution data, engineers time-series features into a
feature store, backfills history, trains and compares nine model approaches,
registers and conditionally promotes the best, generates an hourly 3-day forecast
with hazard alerts and SHAP-style explanations, and serves everything through a
FastAPI backend and a multipage Streamlit dashboard. Hourly and daily GitHub
Actions automate the feature and training pipelines. The whole system runs
locally with **no paid services and no API keys** (keyless Open-Meteo + local
Parquet feature store + local model registry), and swaps to Hopsworks /
OpenWeather via environment variables.

The metrics in this report come from a reproducible **sample-mode run on
synthetic data** and are clearly labelled as such; production metrics require a
real-data backfill (marked `Pending`).

## Problem statement

Urban air quality in megacities such as Karachi frequently reaches unhealthy
levels. Residents and authorities benefit from short-term AQI forecasts to plan
outdoor activity and issue advisories. The goal is a reproducible, automated,
explainable 72-hour AQI forecasting system.

## Objectives

Collect data → store raw + processed → engineer features → backfill history →
train/compare models → register the best → forecast 72h → dashboard + API →
explain → alert → automate (hourly/daily) → test/document/deploy.

## Architecture

Four stages — providers → feature pipeline → training pipeline → forecasting app
— glued by a feature store and model registry, automated by GitHub Actions.
See [`architecture.md`](architecture.md). Abstraction interfaces make providers,
feature store, registry, models, and alert channels swappable without rewrites.

## Data sources

- **Open-Meteo** (default, keyless, CC BY 4.0): air-quality (PM2.5/PM10/CO/NO/
  NO2/O3/SO2/NH3) + weather forecast + ERA5 weather archive.
- **OpenWeather** (alternative, API key): Air Pollution + forecast/current
  weather.
- **Sample provider** (offline): deterministic synthetic data for demos/tests.

## Data collection process

Providers implement `fetch_current` / `fetch_historical` (and `fetch_forecast`
for weather) behind Protocols, each with timeout, retry+backoff, rate-limit
handling, response validation, empty-response handling, logging, and source
attribution. Responses are normalised to a canonical UTC-timestamped schema
([`data_dictionary.md`](data_dictionary.md)).

## Data quality analysis

`data/validation.py` checks required columns, primary-key uniqueness,
chronological order, physical bounds, missing-value rates, and hourly-timestamp
continuity, producing a structured report (hard-fail on critical issues, warn on
recoverable ones). The backfill writes a missing-data report. `Pending: run the
EDA notebook on backfilled production data and summarise real data-quality
findings here.`

## Feature engineering

~248 leakage-free features: current pollutants + weather; calendar (hour, dow,
month, season, weekend) and cyclical (sin/cos) encodings; approximate solar
geometry; backward lags (1/3/6/12/24/48/72h); trailing rolling stats
(3–72h × mean/min/max/std/median); change/acceleration; and interactions
(temp×humidity, wind×PM2.5, rain×PM2.5, hour×NO2, …). All are computable at
prediction time. The target is the locally-computed US EPA AQI, built into
`target_aqi_t_plus_1..72` for direct multi-horizon forecasting.

## Historical backfill

`scripts/backfill_historical_data.py` runs the feature logic over a date range in
configurable batches: resumable (skips completed batches), idempotent/deduped
(upsert on `city_id+timestamp`), with failed-interval retries, a written summary,
a missing-data report, dry-run mode, and a `--sample` fallback when live history
is unavailable.

## EDA findings

The EDA notebook (`notebooks/01_eda.ipynb`) uses `src` functions to analyse AQI
and pollutant distributions, missing data, seasonal/monthly/weekly/hourly
patterns, weekend vs weekday, weather–AQI correlations, lag/auto-correlation,
and extreme episodes. On the **synthetic sample** it reproduces the intended
diurnal (rush-hour) and seasonal (winter-worse) structure and PM2.5-dominated
AQI. `Pending: replace with findings from production data.`

## Model experiments

Nine approaches were compared (direct multi-horizon, 72 outputs):
four baselines (persistence, seasonal-naive, rolling-average, hour-of-day
climatology), Ridge, ElasticNet, RandomForest, HistGradientBoosting, and an
optional TensorFlow MLP. Preprocessing (median impute + optional scaling) is
fit inside a pipeline on training data only. Linear/RF/TF use native
multi-output; HGB is wrapped per-horizon. Hyperparameter search (when enabled)
uses `TimeSeriesSplit` with a gap.

## Evaluation methodology

Chronological split train → gap → validation → gap → test (never shuffled).
Selection on **validation** MAE (tie-break: hazard-period MAE, then artifact
size). The **test** split is reserved for final reporting only. Metrics: RMSE,
MAE, R², sMAPE/MAPE, median AE, bias, per-horizon and per-category error, and
hazard-period MAE.

## Model comparison (sample-mode, synthetic data)

Direct 72-horizon, no hyperparameter search. Split: train 3310 / val 709 /
test 566.

| Model | Val MAE | Val RMSE | Val R² | Test MAE | Hazard MAE |
| ----- | ------: | -------: | -----: | -------: | ---------: |
| **elastic_net (selected)** | **20.96** | **25.46** | **0.092** | **19.36** | 37.55 |
| rolling_average | 22.13 | 27.10 | −0.029 | 22.45 | 51.94 |
| random_forest | 24.88 | 30.03 | −0.264 | 26.56 | 29.95 |
| ridge | 26.04 | 32.29 | −0.461 | 26.99 | 30.34 |
| hist_gradient_boosting | 26.99 | 32.64 | −0.493 | 28.59 | 26.53 |
| persistence | 29.39 | 36.82 | −0.900 | 29.38 | 51.77 |
| seasonal_naive | 29.93 | 37.35 | −0.956 | 29.43 | 53.10 |
| hour_of_day | 43.78 | 49.56 | −2.443 | 47.84 | 9.14 |
| tensorflow | optional extra — not installed in this run |

> These numbers validate the **pipeline**, not real-world accuracy. The low /
> negative R² is expected: the synthetic series is dominated by an
> autoregressive random component that is hard to predict over 1–72h, so most
> models barely beat the mean at long horizons. `Pending: production metrics.`

## Winning model

**ElasticNet** — lowest validation MAE (20.96), registered as
`aqi_forecast_karachi_pakistan` v1 and promoted to `approved` (no incumbent
existed). Selection is transparent and logged; promotion is gated by MAE
improvement, hazard-MAE tolerance, and a smoke test.

## Explainability results

`src/pearls_aqi/explainability/` provides global permutation importance and
local occlusion attribution for any horizon, plus a plain-language summary
(SHAP is used when installed). Importances are labelled as reflecting **model
behaviour, not causation.** `Pending: attach representative importance charts
from a production run.`

## Web application

- **FastAPI** (`api/`): `/health`, `/ready`, `/api/v1/{current,forecast,history,
  model,model/metrics,explanations}`, `POST /api/v1/predict`, OpenAPI at `/docs`.
- **Streamlit** (`app/`): Overview, Three-Day Forecast, Current Pollutants,
  Historical Trends, Model Performance, Explainability, System Status — with an
  AQI colour scale (plus text + icons, never colour alone), timezone selector,
  caching, hazard banners, CSV download, and visible model version / generation
  time.

## Automation

GitHub Actions: `ci.yml` (lint/format/type/test/docker), `feature_pipeline.yml`
(hourly), `training_pipeline.yml` (daily, conditional promotion), `deploy.yml`
(gated deploy). All run without secrets in sample/Open-Meteo mode.

## Testing

pytest suite: unit (AQI, categories, features, lags/rolling/cyclical, targets,
timeutils, providers, alerts, selection, metrics, splits, storage, models),
integration (feature pipeline, storage/registry, training, forecast service,
API), contract (provider/prediction/model schemas), and smoke (end-to-end on
fixtures). External APIs are mocked; no production credentials required. Quality
gates: Ruff, Black, mypy.

## Deployment

Dashboard → Streamlit Community Cloud; API → Render/Railway/Cloud Run (Docker);
storage → Hopsworks; orchestration → GitHub Actions. See
[`deployment.md`](deployment.md).

## Limitations

See [`limitations.md`](limitations.md): horizon skill decay, synthetic sample
data, provider free-tier caps, approximate intervals, future-weather not yet a
model input, single-node local backends.

## Ethical & health considerations

Health-impacting outputs carry a prominent disclaimer; no causal claims are made
from feature importance; synthetic data is always labelled; secrets are never
committed; the model is not for medical/regulatory use.

## Future improvements

Fold horizon-aligned future weather into features; probabilistic/quantile
forecasts with calibrated intervals; per-pollutant multi-target models;
drift-triggered retraining; more cities; API caching.

## Reproduction instructions

```bash
make venv && make install
cp config/config.example.yaml config/config.yaml && cp .env.example .env
make seed            # or: python scripts/backfill_historical_data.py ... for real data
make train           # trains, compares, registers, promotes
make forecast        # 72h forecast + alerts
make api             # http://localhost:8000/docs
make dashboard       # http://localhost:8501
make test            # full test suite
```

To reproduce the exact sample-mode table above:
`python scripts/seed_sample_data.py --city Karachi --days 200 && python
scripts/run_training_pipeline.py --city Karachi --no-tune`.
