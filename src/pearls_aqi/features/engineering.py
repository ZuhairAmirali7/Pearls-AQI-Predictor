"""Feature engineering.

Design rule enforced throughout: **every feature is computable at prediction
time from information available up to and including the current hour** — no
future leakage. Lags/rolling windows look backward; calendar/solar features
depend only on the timestamp and location.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pearls_aqi.aqi import aqi_from_concentrations
from pearls_aqi.config.models import FeaturesConfig
from pearls_aqi.data.domain import Location
from pearls_aqi.data.schemas import (
    METADATA_COLUMNS,
    SUBINDEX_COLUMNS,
)
from pearls_aqi.utils.logging import get_logger

logger = get_logger(__name__)

# Columns never used as model inputs.
_NON_FEATURE_COLUMNS = set(
    METADATA_COLUMNS
    + ["timestamp", "dominant_pollutant", "weather_condition", "data_source", "ingested_at"]
)


def merge_observations(air_quality: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Merge pollution + weather frames on the hourly UTC timestamp."""
    if air_quality.empty and weather.empty:
        return pd.DataFrame()
    aq = air_quality.copy()
    wx = weather.copy()
    for df in (aq, wx):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h")
    # Drop provider-attribution columns before merge to avoid clashes.
    aq = aq.drop(columns=[c for c in ("data_source",) if c in aq.columns], errors="ignore")
    wx = wx.drop(columns=[c for c in ("data_source",) if c in wx.columns], errors="ignore")
    if aq.empty:
        return wx
    if wx.empty:
        return aq
    merged = pd.merge(aq, wx, on="timestamp", how="outer")
    return merged.sort_values("timestamp").reset_index(drop=True)


def compute_aqi_columns(df: pd.DataFrame, standard: str = "us_epa") -> pd.DataFrame:
    """(Re)compute ``aqi``, ``dominant_pollutant``, and sub-indices per row."""
    df = df.copy()
    aqis: list[float | None] = []
    doms: list[str | None] = []
    subs: dict[str, list[float | None]] = {c: [] for c in SUBINDEX_COLUMNS}
    pollutants = ("pm2_5", "pm10", "o3", "no2", "so2", "co")
    for _, row in df.iterrows():
        res = aqi_from_concentrations({p: row.get(p) for p in pollutants}, standard=standard)
        aqis.append(res.aqi)
        doms.append(res.dominant_pollutant)
        for p in pollutants:
            subs[f"aqi_{p}"].append(res.sub_indices.get(p))
    df["aqi"] = aqis
    df["dominant_pollutant"] = doms
    for col, values in subs.items():
        df[col] = values
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_month"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    # Meteorological season (Northern Hemisphere): 0=winter..3=autumn.
    df["season"] = (ts.dt.month % 12 // 3).astype(int)
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    if "wind_direction" in df.columns:
        rad = np.deg2rad(df["wind_direction"].astype(float))
        df["wind_dir_sin"] = np.sin(rad)
        df["wind_dir_cos"] = np.cos(rad)
    return df


def add_solar_features(df: pd.DataFrame, location: Location | None) -> pd.DataFrame:
    """Approximate sunrise/sunset proximity from latitude, longitude, day-of-year.

    Uses a standard solar-geometry approximation (ignoring the equation of time)
    — accurate to within minutes, and fully leakage-free (depends only on the
    timestamp + location). Guarded so it can never crash the pipeline.
    """
    df = df.copy()
    if location is None:
        return df
    try:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        doy = ts.dt.dayofyear.to_numpy()
        frac_hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60.0
        lat = np.deg2rad(location.latitude)
        decl = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)
        cos_h = -np.tan(lat) * np.tan(decl)
        cos_h = np.clip(cos_h, -1.0, 1.0)
        half_day = np.rad2deg(np.arccos(cos_h)) / 15.0  # hours
        solar_noon_utc = 12.0 - location.longitude / 15.0
        sunrise = solar_noon_utc - half_day
        sunset = solar_noon_utc + half_day
        df["daylight_hours"] = 2 * half_day
        df["hours_since_sunrise"] = frac_hour - sunrise
        df["hours_to_sunset"] = sunset - frac_hour
        df["is_daytime"] = ((frac_hour >= sunrise) & (frac_hour <= sunset)).astype(int)
    except Exception as exc:
        logger.warning("Solar feature computation skipped: %s", exc)
    return df


def _grouped(df: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    return (
        df.groupby("city_id", group_keys=False)
        if "city_id" in df.columns
        else df.groupby(lambda _: 0, group_keys=False)
    )


def add_lag_features(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    new_cols: dict[str, pd.Series] = {}
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            new_cols[f"{col}_lag_{lag}h"] = _grouped(df)[col].shift(lag)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1) if new_cols else df


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
    stats: list[str],
    min_fraction: float = 0.5,
) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    has_city = "city_id" in df.columns
    new_cols: dict[str, pd.Series] = {}
    for col in columns:
        if col not in df.columns:
            continue
        for w in windows:
            min_periods = max(1, int(np.ceil(w * min_fraction)))
            if has_city:
                roll = _grouped(df)[col].rolling(window=w, min_periods=min_periods)
                for stat in stats:
                    new_cols[f"{col}_roll_{stat}_{w}h"] = getattr(roll, stat)().reset_index(
                        level=0, drop=True
                    )
            else:
                roll = df[col].rolling(window=w, min_periods=min_periods)
                for stat in stats:
                    new_cols[f"{col}_roll_{stat}_{w}h"] = getattr(roll, stat)()
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1) if new_cols else df


def add_change_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    if "aqi" in df.columns:
        df["aqi_change_1h"] = _grouped(df)["aqi"].diff(1)
        df["aqi_pct_change_1h"] = (
            _grouped(df)["aqi"].pct_change(1, fill_method=None).replace([np.inf, -np.inf], np.nan)
        )
        # Acceleration = change of the change (second difference), per city.
        df["aqi_acceleration"] = _grouped(df)["aqi_change_1h"].diff(1)
    for col in ("pm2_5", "pm10", "temperature", "pressure", "humidity", "wind_speed"):
        if col in df.columns:
            df[f"{col}_change_1h"] = _grouped(df)[col].diff(1)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"temperature", "humidity"} <= set(df.columns):
        df["temp_x_humidity"] = df["temperature"] * df["humidity"]
    if {"wind_speed", "pm2_5"} <= set(df.columns):
        df["wind_x_pm25"] = df["wind_speed"] * df["pm2_5"]
    if {"pressure_change_1h", "aqi"} <= set(df.columns):
        df["pressure_change_x_aqi"] = df["pressure_change_1h"] * df["aqi"]
    if {"rain", "pm2_5"} <= set(df.columns):
        df["rain_indicator"] = (df["rain"].fillna(0) > 0).astype(int)
        df["rain_x_pm25"] = df["rain_indicator"] * df["pm2_5"]
    if {"hour", "no2"} <= set(df.columns):
        df["hour_x_no2"] = df["hour"] * df["no2"]
    return df


def build_features(
    df: pd.DataFrame,
    features_config: FeaturesConfig,
    location: Location | None = None,
    recompute_aqi: bool = False,
    aqi_standard: str = "us_epa",
) -> pd.DataFrame:
    """Run the full feature-engineering chain on a merged observation frame."""
    if df.empty:
        return df
    out = df.sort_values("timestamp").reset_index(drop=True)
    if recompute_aqi or "aqi" not in out.columns:
        out = compute_aqi_columns(out, standard=aqi_standard)

    out = add_time_features(out)
    out = add_cyclical_features(out)
    out = add_solar_features(out, location)
    out = add_lag_features(out, features_config.lag_pollutants, features_config.lag_hours)
    out = add_rolling_features(
        out,
        features_config.lag_pollutants,
        features_config.rolling_windows,
        features_config.rolling_stats,
        features_config.rolling_min_fraction,
    )
    out = add_change_features(out)
    out = add_interaction_features(out)
    logger.info("Built %d features on %d rows.", out.shape[1], out.shape[0])
    return out


def select_feature_columns(df: pd.DataFrame, target_columns: list[str] | None = None) -> list[str]:
    """Numeric model-input columns: excludes metadata, targets, and identifiers."""
    exclude = set(_NON_FEATURE_COLUMNS)
    if target_columns:
        exclude.update(target_columns)
    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.startswith("target_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols
