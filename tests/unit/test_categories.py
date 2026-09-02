"""Unit tests for AQI category classification."""

from __future__ import annotations

import pytest

from pearls_aqi.data.domain import AQICategory, categorize_aqi


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "Good"),
        (50, "Good"),
        (51, "Moderate"),
        (100, "Moderate"),
        (101, "Unhealthy for Sensitive Groups"),
        (150, "Unhealthy for Sensitive Groups"),
        (151, "Unhealthy"),
        (200, "Unhealthy"),
        (201, "Very Unhealthy"),
        (300, "Very Unhealthy"),
        (301, "Hazardous"),
        (500, "Hazardous"),
        (750, "Hazardous"),
    ],
)
def test_category_boundaries(value, expected):
    assert categorize_aqi(value).value == expected


def test_nan_raises():
    with pytest.raises(ValueError):
        categorize_aqi(float("nan"))


def test_none_raises():
    with pytest.raises(ValueError):
        categorize_aqi(None)  # type: ignore[arg-type]


def test_category_has_ui_attributes():
    for category in AQICategory:
        assert category.color.startswith("#")
        assert category.icon
        assert isinstance(category.health_message, str) and category.health_message
