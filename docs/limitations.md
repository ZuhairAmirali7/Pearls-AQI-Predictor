# Known Limitations & Assumptions

Honesty is a first-class requirement of this project. This document lists what
the system does **not** do well and the assumptions it makes.

## Health / safety

- **Not a substitute for official warnings or medical advice.** Forecasts are
  model estimates for information only. Do not use them for health, safety, or
  emergency decisions. See [`model_card.md`](model_card.md).

## Data

- **Sample data is synthetic.** `data/sample/` and anything produced by
  `--sample` / sample mode is deterministic synthetic data (`data_source =
  sample`), never real observations. It exists so the project runs offline.
- **Provider free-tier limits.**
  - Open-Meteo air-quality history goes back to ~2022; the ERA5 weather archive
    has a few-day processing lag near "now".
  - OpenWeather's free tier does **not** expose bulk historical hourly weather,
    so historical backfill requires Open-Meteo (or a supplied CSV). The
    OpenWeather weather provider raises a clear error for `fetch_historical`.
  - OpenWeather's 5-day forecast is 3-hourly and is interpolated to hourly.
- **Backfill depth** is bounded by provider availability and rate limits; large
  ranges are batched and may take time.

## AQI computation

- US EPA AQI is computed locally from concentrations. Gaseous pollutants are
  converted from µg/m³ to EPA units assuming standard conditions (25 °C, 1 atm),
  which introduces small error versus true in-situ conditions.
- O3 above 0.200 ppm (8-hr) and SO2 above 304 ppb (1-hr) use averaging rules not
  modelled here; sub-indices are clamped to the top of the implemented table.
- Overall AQI above 500 is clamped to 500.

## Modelling

- **Skill degrades with horizon** — a 72h forecast is far less certain than a 1h
  one; error-by-horizon is reported and shown in the dashboard.
- **Extreme episodes are under-represented**, so hazardous-period error is higher
  and noisier.
- **Prediction intervals are approximate** (scaled per-horizon validation MAE),
  not calibrated coverage. Treat them as rough uncertainty, not guarantees.
- **Future weather is fetched but not yet a model input.** The direct model
  conditions on observed history up to time `t`. The prediction service fetches
  the future hourly weather forecast (used for context/display); folding
  horizon-aligned future weather into the feature vector is a planned
  improvement.
- **Sample-mode metrics are not real-world accuracy.** The synthetic series is
  intentionally hard (strong random AR component), yielding low/negative R².
  Real data typically gives much better short-horizon skill.

## Infrastructure

- The default local Parquet feature store and filesystem registry are
  single-node and not concurrency-safe for many writers; use Hopsworks for
  multi-consumer/cloud setups.
- In GitHub Actions the local store is ephemeral; scheduled runs are only
  persistent when Hopsworks is configured.
- TensorFlow, SHAP, and Hopsworks are optional extras; when absent the system
  degrades gracefully (TF model skipped, permutation-importance fallback, local
  backends).

## Scope

- One forecast target (overall AQI). Per-pollutant multi-target forecasting and
  probabilistic/quantile models are future work.
- Alerts are dashboard/log by default; email/webhook are opt-in and best-effort.
