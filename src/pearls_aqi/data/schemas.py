"""Canonical column schemas and physical bounds.

Every layer (providers, features, storage, validation, API) agrees on these
column names and units so data flows without ad-hoc renaming. All concentrations
are µg/m³; temperatures °C; pressure hPa; wind m/s and degrees; timestamps UTC.
"""

from __future__ import annotations

# Primary key for every stored observation row.
PRIMARY_KEY: list[str] = ["city_id", "timestamp"]
EVENT_TIME: str = "timestamp"

# Provenance columns attached to every record.
METADATA_COLUMNS: list[str] = [
    "city_id",
    "city",
    "country",
    "latitude",
    "longitude",
    "timezone",
    "data_source",
    "ingested_at",
    "feature_pipeline_version",
]

# Pollutant concentrations (µg/m³).
POLLUTANT_COLUMNS: list[str] = [
    "pm2_5",
    "pm10",
    "co",
    "no",
    "no2",
    "o3",
    "so2",
    "nh3",
]

# AQI columns derived by the feature pipeline.
AQI_COLUMNS: list[str] = ["aqi", "dominant_pollutant"]
# Per-pollutant AQI sub-indices (optional; prefixed to avoid clashing with
# raw concentration columns).
SUBINDEX_COLUMNS: list[str] = [f"aqi_{p}" for p in ("pm2_5", "pm10", "o3", "no2", "so2", "co")]

# Weather observations.
WEATHER_COLUMNS: list[str] = [
    "temperature",
    "feels_like",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "wind_gust",
    "precipitation",
    "rain",
    "cloud_cover",
    "visibility",
    "dew_point",
    "weather_condition",
    "boundary_layer_height",
]

# Numeric columns eligible for validation / feature engineering.
NUMERIC_OBSERVATION_COLUMNS: list[str] = (
    POLLUTANT_COLUMNS + ["aqi"] + [c for c in WEATHER_COLUMNS if c != "weather_condition"]
)

# Physical plausibility bounds (inclusive). Values outside these are flagged by
# the validator. ``None`` means unbounded on that side.
PHYSICAL_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "pm2_5": (0.0, 2000.0),
    "pm10": (0.0, 5000.0),
    "co": (0.0, 100000.0),
    "no": (0.0, 5000.0),
    "no2": (0.0, 5000.0),
    "o3": (0.0, 2000.0),
    "so2": (0.0, 5000.0),
    "nh3": (0.0, 5000.0),
    "aqi": (0.0, 500.0),
    "temperature": (-60.0, 60.0),
    "feels_like": (-70.0, 70.0),
    "humidity": (0.0, 100.0),
    "pressure": (850.0, 1100.0),
    "wind_speed": (0.0, 120.0),
    "wind_direction": (0.0, 360.0),
    "wind_gust": (0.0, 150.0),
    "precipitation": (0.0, 500.0),
    "rain": (0.0, 500.0),
    "cloud_cover": (0.0, 100.0),
    "visibility": (0.0, 100000.0),
    "dew_point": (-60.0, 50.0),
    "boundary_layer_height": (0.0, 6000.0),
}

# The target column produced by the feature pipeline.
TARGET_BASE_COLUMN: str = "aqi"


def observation_columns() -> list[str]:
    """Full ordered column list for a merged observation row."""
    return (
        METADATA_COLUMNS
        + [EVENT_TIME]
        + POLLUTANT_COLUMNS
        + AQI_COLUMNS
        + SUBINDEX_COLUMNS
        + WEATHER_COLUMNS
    )
