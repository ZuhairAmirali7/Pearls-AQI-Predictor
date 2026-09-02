# Data Dictionary

Canonical column definitions. Source of truth: `src/pearls_aqi/data/schemas.py`
and `src/pearls_aqi/features/engineering.py`. Units: concentrations µg/m³,
temperature °C, pressure hPa, wind m/s and degrees, timestamps UTC.

## Provenance / metadata (every stored row)

| Column | Type | Description |
| ------ | ---- | ----------- |
| `city_id` | str | Slug primary-key part, e.g. `karachi_pakistan`. |
| `city` | str | City name. |
| `country` | str | Country name. |
| `latitude` | float | Degrees north. |
| `longitude` | float | Degrees east. |
| `timezone` | str | IANA timezone (display only). |
| `data_source` | str | Provider that produced the row (`openmeteo`/`openweather`/`sample`). |
| `ingested_at` | datetime(UTC) | When the row was ingested. |
| `feature_pipeline_version` | str | Semantic version of the feature code. |
| `timestamp` | datetime(UTC) | **Event time** — the observation hour. Primary-key part. |

Primary key: `(city_id, timestamp)`. Event-time field: `timestamp`.

## Pollutant concentrations (µg/m³)

| Column | Description |
| ------ | ----------- |
| `pm2_5` | Fine particulate matter (< 2.5 µm). |
| `pm10` | Coarse particulate matter (< 10 µm). |
| `co` | Carbon monoxide. |
| `no` | Nitric oxide. |
| `no2` | Nitrogen dioxide. |
| `o3` | Ozone. |
| `so2` | Sulphur dioxide. |
| `nh3` | Ammonia. |

## AQI columns (computed)

| Column | Description |
| ------ | ----------- |
| `aqi` | Overall US EPA AQI (0–500), the **prediction target**. Computed locally from concentrations (`aqi.py`). |
| `dominant_pollutant` | Pollutant driving the overall AQI. |
| `aqi_pm2_5`, `aqi_pm10`, `aqi_o3`, `aqi_no2`, `aqi_so2`, `aqi_co` | Per-pollutant AQI sub-indices. |

The AQI target is **locally computed** (`data.target_source: local_computed`),
not taken from the provider's own index, so it is consistent and provider-
independent. PM2.5 uses the 2024 EPA breakpoints.

## Weather columns

| Column | Unit | Description |
| ------ | ---- | ----------- |
| `temperature` | °C | Air temperature (2 m). |
| `feels_like` | °C | Apparent temperature. |
| `humidity` | % | Relative humidity. |
| `pressure` | hPa | Surface pressure. |
| `wind_speed` | m/s | Wind speed (10 m). |
| `wind_direction` | ° | Wind direction (0–360). |
| `wind_gust` | m/s | Wind gust. |
| `precipitation` | mm | Total precipitation. |
| `rain` | mm | Rainfall. |
| `cloud_cover` | % | Cloud cover. |
| `visibility` | m | Horizontal visibility. |
| `dew_point` | °C | Dew point. |
| `weather_condition` | str | Coarse label (Clear/Clouds/Rain/…). |
| `boundary_layer_height` | m | Planetary boundary-layer height (when available). |

## Engineered features (`build_features`)

All engineered features are **computable at prediction time** (no future
leakage): lags/rolling look backward; calendar/solar features depend only on the
timestamp + location.

| Family | Columns | Notes |
| ------ | ------- | ----- |
| Time | `hour`, `day_of_week`, `day_of_month`, `month`, `day_of_year`, `week_of_year`, `is_weekend`, `season` | Calendar features. |
| Cyclical | `sin_hour`/`cos_hour`, `sin_day_of_week`/`cos_day_of_week`, `sin_month`/`cos_month`, `wind_dir_sin`/`wind_dir_cos` | Periodic encodings. |
| Solar | `daylight_hours`, `hours_since_sunrise`, `hours_to_sunset`, `is_daytime` | Approx. solar geometry. |
| Lag | `<col>_lag_{1,3,6,12,24,48,72}h` for `aqi, pm2_5, pm10, o3, no2` | Backward shifts per city. |
| Rolling | `<col>_roll_{mean,min,max,std,median}_{3,6,12,24,48,72}h` | Trailing windows (min-fraction gated). |
| Change | `aqi_change_1h`, `aqi_pct_change_1h`, `aqi_acceleration`, `<col>_change_1h` for pm2_5/pm10/temperature/pressure/humidity/wind_speed | First/second differences. |
| Interaction | `temp_x_humidity`, `wind_x_pm25`, `pressure_change_x_aqi`, `rain_indicator`, `rain_x_pm25`, `hour_x_no2` | Domain interactions. |

## Targets (`features/targets.py`)

| Column | Description |
| ------ | ----------- |
| `target_aqi_t_plus_{1..72}` | AQI observed `h` hours ahead (direct multi-horizon). Rows whose targets extend past the series end are `NaN` and excluded from training (never imputed). |

## Prediction output (`model_predictions` group)

`timestamp`, `horizon_hour`, `predicted_aqi`, `lower_bound`, `upper_bound`,
`category`, `dominant_pollutant`, plus `city_id`, `city`, `model_name`,
`model_version`, `generated_at`.

## Physical bounds

Validation flags values outside plausible physical ranges (see
`PHYSICAL_BOUNDS` in `schemas.py`), e.g. AQI ∈ [0, 500], humidity ∈ [0, 100] %,
pressure ∈ [850, 1100] hPa. Out-of-range values raise warnings (not hard
failures) and are surfaced in the validation report.
