"""Tests for the Streamlit dashboard's pure helpers and data-access layer.

No Streamlit server is started. The deterministic AQI-UI helpers are asserted
exactly; the data-access functions are checked for *graceful* behaviour (they
must return a dict and never raise, whether or not default data exists).
"""

from __future__ import annotations

import pytest
from app.components import aqi as aqi_ui

from pearls_aqi.data.domain import AQICategory

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "value,expected",
    [
        (25, AQICategory.GOOD),
        (75, AQICategory.MODERATE),
        (125, AQICategory.UNHEALTHY_SENSITIVE),
        (175, AQICategory.UNHEALTHY),
        (250, AQICategory.VERY_UNHEALTHY),
        (400, AQICategory.HAZARDOUS),
    ],
)
def test_category_of(value, expected):
    assert aqi_ui.category_of(value) == expected


def test_aqi_color_and_badge():
    assert aqi_ui.aqi_color(25) == AQICategory.GOOD.color
    badge = aqi_ui.category_badge(175)
    # Badge must convey the level with text, not colour alone.
    assert "Unhealthy" in badge
    assert aqi_ui.category_of(None) is None


def test_health_message_nonempty():
    assert aqi_ui.health_message(175)


def test_services_graceful():
    """Data-access functions return dicts and never raise (graceful states)."""
    import app.services as services

    assert isinstance(services.list_cities(), list)
    assert services.mode() in {"direct", "api"}
    for fn in (services.get_forecast, services.get_current, services.get_system_status):
        result = fn("Karachi")
        assert isinstance(result, dict)
