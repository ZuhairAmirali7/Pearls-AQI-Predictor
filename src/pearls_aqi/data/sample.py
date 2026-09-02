"""Deterministic synthetic observation generator (SAMPLE MODE).

Produces realistic-looking hourly weather + pollution data with diurnal and
seasonal structure so the whole system can run offline with no API access.

⚠️  This data is **synthetic**. It is clearly marked with ``data_source =
'sample'`` and must never be presented as real observations. Its only purpose is
to make the project runnable and testable without network/credentials.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from pearls_aqi.aqi import aqi_from_concentrations
from pearls_aqi.data.domain import Location
from pearls_aqi.data.schemas import METADATA_COLUMNS
from pearls_aqi.utils.timeutils import ensure_utc, now_utc, utc_floor_hour

SAMPLE_SOURCE = "sample"


def generate_sample_observations(
    location: Location,
    start: datetime,
    end: datetime,
    seed: int = 42,
    feature_pipeline_version: str = "1.0.0",
) -> pd.DataFrame:
    """Generate deterministic hourly observations between ``start`` and ``end``.

    The seed makes output reproducible; it is derived from ``seed`` and the city
    so different cities look different but each city is stable across runs.
    """
    start_h = utc_floor_hour(start)
    end_h = utc_floor_hour(end)
    if end_h < start_h:
        raise ValueError("end must be >= start")

    index = pd.date_range(start_h, end_h, freq="h", tz="UTC")
    n = len(index)
    city_seed = (seed + abs(hash(location.city_id))) % (2**31)
    rng = np.random.default_rng(city_seed)

    hour = index.hour.to_numpy()
    doy = index.dayofyear.to_numpy()
    # Seasonal factor: pollution peaks in winter (Dec–Jan) in South Asia.
    season = np.cos(2 * np.pi * (doy - 15) / 365.25)  # +1 around mid-Jan
    # Diurnal factor: morning (8h) and evening (20h) traffic peaks, cleaner midday.
    diurnal = (
        0.6 * np.exp(-0.5 * ((hour - 8) / 2.5) ** 2)
        + 0.8 * np.exp(-0.5 * ((hour - 20) / 3.0) ** 2)
        + 0.2
    )

    # --- Weather ---
    temp_season = 28 + 8 * np.sin(2 * np.pi * (doy - 100) / 365.25)  # warm most of year
    temp_diurnal = 5 * np.sin(2 * np.pi * (hour - 15) / 24)  # peak mid-afternoon
    temperature = temp_season + temp_diurnal + rng.normal(0, 1.2, n)
    humidity = np.clip(70 - 0.9 * (temperature - 28) + rng.normal(0, 6, n), 5, 100)
    pressure = 1009 + 4 * np.sin(2 * np.pi * doy / 365.25) + rng.normal(0, 2, n)
    wind_speed = np.clip(_ar1(rng, n, 2.8, 0.7, 1.0), 0.1, 25)
    wind_direction = (180 + 60 * np.sin(2 * np.pi * doy / 365.25) + rng.normal(0, 40, n)) % 360
    wind_gust = wind_speed * rng.uniform(1.1, 1.8, n)
    cloud_cover = np.clip(
        40 + 30 * np.sin(2 * np.pi * (doy - 180) / 365.25) + rng.normal(0, 15, n), 0, 100
    )
    rain_event = (rng.random(n) < np.clip(cloud_cover / 100 * 0.15, 0, 0.4)).astype(float)
    precipitation = rain_event * rng.gamma(2.0, 1.5, n)
    dew_point = temperature - (100 - humidity) / 5.0
    feels_like = temperature + 0.3 * (humidity / 100) * (temperature > 27) - 0.2 * wind_speed
    visibility = np.clip(
        10000 - 60 * (diurnal * (1 + season)) * 30 + rng.normal(0, 800, n), 300, 20000
    )
    boundary_layer_height = np.clip(
        1200 + 600 * np.sin(2 * np.pi * (hour - 15) / 24) + rng.normal(0, 150, n), 100, 3000
    )

    # --- Pollution (µg/m³) ---
    # Base level with a seasonal swing (winter dirtier), modulated by the
    # diurnal traffic profile recentred so midday stays elevated (realistic for
    # a dense South-Asian megacity rather than dropping near zero off-peak).
    base_pm25 = 55 + 22 * season  # ~33 µg/m³ summer, ~77 winter
    pm2_5 = base_pm25 * (0.7 + diurnal)
    pm2_5 = pm2_5 * (1.0 / (1.0 + 0.12 * wind_speed))  # wind disperses
    pm2_5 = pm2_5 * (1 - 0.4 * rain_event)  # rain scavenges
    pm2_5 = np.clip(_ar1_multiplicative(rng, pm2_5, 0.6), 2, 900)
    pm10 = np.clip(pm2_5 * rng.uniform(1.4, 2.1, n) + rng.normal(0, 5, n), 3, 1200)
    no2 = np.clip(18 * diurnal * (1 + 0.4 * season) + rng.normal(0, 4, n), 1, 400)
    no = np.clip(no2 * rng.uniform(0.3, 0.8, n), 0, 300)
    so2 = np.clip(8 * diurnal + rng.normal(0, 2, n), 0.5, 200)
    o3 = np.clip(60 * np.exp(-0.5 * ((hour - 15) / 3.5) ** 2) + 10 + rng.normal(0, 6, n), 1, 250)
    co = np.clip(400 * diurnal * (1 + 0.3 * season) + rng.normal(0, 40, n), 50, 8000)
    nh3 = np.clip(12 * diurnal + rng.normal(0, 3, n), 0.5, 200)

    df = pd.DataFrame(
        {
            "timestamp": index,
            "pm2_5": pm2_5,
            "pm10": pm10,
            "co": co,
            "no": no,
            "no2": no2,
            "o3": o3,
            "so2": so2,
            "nh3": nh3,
            "temperature": temperature,
            "feels_like": feels_like,
            "humidity": humidity,
            "pressure": pressure,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "wind_gust": wind_gust,
            "precipitation": precipitation,
            "rain": precipitation,
            "cloud_cover": cloud_cover,
            "visibility": visibility,
            "dew_point": dew_point,
            "weather_condition": np.where(
                rain_event > 0, "Rain", np.where(cloud_cover > 60, "Clouds", "Clear")
            ),
            "boundary_layer_height": boundary_layer_height,
        }
    )

    df = _attach_aqi(df)
    df = _attach_metadata(df, location, SAMPLE_SOURCE, feature_pipeline_version)
    return df.round(3)


def _ar1(rng: np.random.Generator, n: int, mean: float, phi: float, sigma: float) -> np.ndarray:
    """First-order autoregressive series (adds realistic temporal correlation)."""
    out = np.empty(n)
    out[0] = mean
    noise = rng.normal(0, sigma, n)
    for t in range(1, n):
        out[t] = mean + phi * (out[t - 1] - mean) + noise[t]
    return out


def _ar1_multiplicative(rng: np.random.Generator, base: np.ndarray, phi: float) -> np.ndarray:
    """Apply AR(1) multiplicative noise around a deterministic base signal."""
    n = len(base)
    factor = np.empty(n)
    factor[0] = 1.0
    noise = rng.normal(0, 0.25, n)
    for t in range(1, n):
        factor[t] = np.clip(1.0 + phi * (factor[t - 1] - 1.0) + noise[t], 0.4, 2.2)
    return base * factor


def _attach_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute AQI, dominant pollutant, and sub-indices per row."""
    aqis, doms = [], []
    sub_cols: dict[str, list[float | None]] = {}
    for _, row in df.iterrows():
        res = aqi_from_concentrations(
            {p: row.get(p) for p in ("pm2_5", "pm10", "o3", "no2", "so2", "co")}
        )
        aqis.append(res.aqi)
        doms.append(res.dominant_pollutant)
        for p, v in res.sub_indices.items():
            sub_cols.setdefault(f"aqi_{p}", []).append(v)
        for p in ("pm2_5", "pm10", "o3", "no2", "so2", "co"):
            if p not in res.sub_indices:
                sub_cols.setdefault(f"aqi_{p}", []).append(None)
    df = df.copy()
    df["aqi"] = aqis
    df["dominant_pollutant"] = doms
    for col, values in sub_cols.items():
        df[col] = values
    return df


def _attach_metadata(
    df: pd.DataFrame, location: Location, source: str, version: str
) -> pd.DataFrame:
    df = df.copy()
    df["city_id"] = location.city_id
    df["city"] = location.city
    df["country"] = location.country
    df["latitude"] = location.latitude
    df["longitude"] = location.longitude
    df["timezone"] = location.timezone
    df["data_source"] = source
    df["ingested_at"] = ensure_utc(now_utc())
    df["feature_pipeline_version"] = version
    # Order metadata first for readability.
    ordered = METADATA_COLUMNS + [c for c in df.columns if c not in METADATA_COLUMNS]
    return df[ordered]
