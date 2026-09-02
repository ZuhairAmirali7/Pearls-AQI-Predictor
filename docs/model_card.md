# Model Card — Pearls AQI Predictor

> **Health disclaimer.** This model produces short-term AQI **estimates for
> informational purposes only**. It **does not replace official government
> air-quality warnings or medical advice.** Do not rely on it for health,
> safety, or emergency decisions.

## Model details

| Field | Value |
| ----- | ----- |
| Model name | `aqi_forecast_<city_id>` (e.g. `aqi_forecast_karachi_pakistan`) |
| Selected type (sample-mode run) | **ElasticNet** (linear, multi-output) |
| Version | 1 (approved) |
| Forecast horizon | 72 hours, hourly |
| Strategy | Direct multi-horizon (one output per hour `t+1 … t+72`) |
| Supported locations | Any city in `config.yaml` (`cities` table). Default: Karachi. |
| Target | Locally-computed **US EPA AQI** (0–500) |
| Explainability | Permutation / occlusion importance (SHAP when installed) |
| Retraining | Daily (GitHub Actions), conditional promotion |

The *type* of the selected model is not fixed — the training pipeline compares
several candidates each run and registers whichever wins on validation MAE. The
values below are from a reproducible **sample-mode** run; rerun on real data for
production metrics.

## Intended use

- Short-term (≤72h) AQI guidance for a configured city, shown in a dashboard/API.
- Not intended for: regulatory reporting, health/medical decisions, legal use,
  or locations without representative training data.

## Training data

- **Sample-mode (this card):** ~200 days of deterministic **synthetic** hourly
  observations (`data_source = sample`). Synthetic — not real measurements.
- **Production:** backfill real history via Open-Meteo (`scripts/backfill_historical_data.py`),
  then retrain. Config default `data.historical_days: 730`.
- Split (sample-mode): chronological **train 3310 / validation 709 / test 566**
  rows, with a 72-hour gap between splits to prevent leakage.

## Features

~248 numeric inputs: current pollutant concentrations + weather, calendar &
cyclical encodings, approximate solar-geometry features, backward lags
(1–72h), trailing rolling statistics (3–72h × mean/min/max/std/median), change
& acceleration, and domain interactions. Every feature is computable at
prediction time (no future leakage). See [`data_dictionary.md`](data_dictionary.md).

## Evaluation metrics (sample-mode; synthetic data)

Direct 72-horizon, no hyperparameter search. **Reproduce for real metrics by
running the pipeline on backfilled production data.**

| Model | Val MAE | Val RMSE | Val R² | Test MAE | Hazard MAE |
| ----- | ------: | -------: | -----: | -------: | ---------: |
| **elastic_net (selected)** | **20.96** | **25.46** | **0.092** | **19.36** | 37.55 |
| rolling_average | 22.13 | 27.10 | −0.029 | 22.45 | 51.94 |
| random_forest | 24.88 | 30.03 | −0.264 | 26.56 | 29.95 |
| ridge | 26.04 | 32.29 | −0.461 | 26.99 | 30.34 |
| hist_gradient_boosting | 26.99 | 32.64 | −0.493 | 28.59 | 26.53 |
| persistence (baseline) | 29.39 | 36.82 | −0.900 | 29.38 | 51.77 |
| seasonal_naive (baseline) | 29.93 | 37.35 | −0.956 | 29.43 | 53.10 |
| hour_of_day (baseline) | 43.78 | 49.56 | −2.443 | 47.84 | 9.14 |
| tensorflow | not installed (optional extra) — auto-skipped |

> **Reading these numbers honestly.** The low / negative R² is expected on this
> *synthetic* series: it is deliberately dominated by an autoregressive random
> component that is hard to predict 1–72h ahead, so most models barely beat the
> mean at long horizons. On real observations, short-horizon skill is typically
> much better (strong persistence). These figures validate the *pipeline*, not
> real-world accuracy. `Pending: insert production metrics after backfilling real
> historical data and retraining.`

## Validation methodology

- Chronological train → gap → validation → gap → test (never shuffled).
- Model selection on the **validation** split (lowest MAE, tie-broken by
  hazard-period MAE, then artifact size). The **test** split is used only for
  final reporting.
- Hyperparameter search (when enabled) uses `TimeSeriesSplit` with a gap.

## Known limitations

- Forecast skill degrades with horizon.
- Extreme/hazardous episodes are under-represented, so hazard-period error is
  higher and more uncertain.
- Prediction intervals are **approximate** (per-horizon validation-MAE scaled),
  not calibrated coverage.
- The v1 feature set conditions on observed history; horizon-aligned future
  weather forecasts are fetched for context but not yet folded into model inputs
  (documented future improvement).
- O3/SO2 sub-indices above certain concentrations are not modelled.

## Ethical & health considerations

- Outputs can influence health behaviour; the disclaimer above is shown in the
  dashboard and API responses/docs.
- Feature importances reflect **model behaviour, not proven causation** — no
  causal claims are made.
- Synthetic sample data is clearly labelled and never presented as real.

## Failure modes

- No approved model / no features → the app returns a clear error and prompts to
  run the pipelines (never a silent wrong answer).
- Feature-schema mismatch → prediction raises `SchemaMismatchError` (retrain).
- Provider outage → retries, then graceful failure; sample mode always works.

## Data-freshness requirements

Hourly feature updates expected; data older than `monitoring.max_data_age_hours`
(default 3h) is flagged stale in the dashboard/monitoring.
