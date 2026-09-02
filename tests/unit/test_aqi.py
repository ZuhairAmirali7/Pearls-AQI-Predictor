"""Unit tests for US EPA AQI conversion."""

from __future__ import annotations

import math

import pytest

from pearls_aqi.aqi import (
    aqi_from_concentrations,
    pollutant_subindex,
    supported_pollutants,
)


def test_pm25_breakpoint_boundaries():
    # 2024 EPA PM2.5 breakpoints.
    assert pollutant_subindex("pm2_5", 9.0) == 50
    assert pollutant_subindex("pm2_5", 9.1) == 51
    assert pollutant_subindex("pm2_5", 35.5) == 101
    assert pollutant_subindex("pm2_5", 0.0) == 0


def test_pm10_breakpoint_boundaries():
    assert pollutant_subindex("pm10", 54) == 50
    assert pollutant_subindex("pm10", 55) == 51
    assert pollutant_subindex("pm10", 154) == 100


def test_overall_aqi_is_max_subindex_and_dominant():
    # PM2.5 sub-index (101) should dominate over cleaner pollutants.
    result = aqi_from_concentrations(
        {"pm2_5": 35.5, "pm10": 40, "o3": 20, "no2": 10, "so2": 5, "co": 200}
    )
    assert result.aqi == max(result.sub_indices.values())
    assert result.dominant_pollutant == "pm2_5"
    assert result.aqi == 101
    assert result.category.value == "Unhealthy for Sensitive Groups"


def test_clamp_above_500():
    # Extremely high PM2.5 clamps at the top of the table (500).
    assert pollutant_subindex("pm2_5", 100000) == 500


def test_missing_pollutant_returns_none_not_zero():
    assert pollutant_subindex("pm2_5", None) is None
    assert pollutant_subindex("pm2_5", float("nan")) is None


def test_missing_pollutant_excluded_from_overall():
    result = aqi_from_concentrations({"pm2_5": None, "pm10": None, "o3": None})
    assert result.aqi is None
    assert result.dominant_pollutant is None
    assert result.sub_indices == {}


def test_negative_concentration_raises():
    with pytest.raises(ValueError):
        pollutant_subindex("pm2_5", -1.0)


def test_gas_unit_conversion_sanity():
    # A modest CO concentration (µg/m³) should map into the Good band after
    # conversion to ppm (well below the 4.4 ppm boundary).
    sub = pollutant_subindex("co", 500.0)
    assert sub is not None
    assert 0 <= sub <= 50


def test_unknown_pollutant_raises():
    with pytest.raises(ValueError):
        pollutant_subindex("plutonium", 1.0)


def test_supported_pollutants_cover_required_set():
    supported = set(supported_pollutants())
    assert {"pm2_5", "pm10", "o3", "no2", "so2", "co"} <= supported


def test_only_us_epa_standard_implemented():
    with pytest.raises(ValueError):
        aqi_from_concentrations({"pm2_5": 10}, standard="uk_daqi")


def test_result_as_dict_is_serialisable():
    result = aqi_from_concentrations({"pm2_5": 50})
    d = result.as_dict()
    assert d["aqi"] is not None and not math.isnan(d["aqi"])
    assert d["category"] is not None
