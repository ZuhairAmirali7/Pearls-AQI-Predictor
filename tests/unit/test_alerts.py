"""Unit tests for hazard-alert detection and dispatch."""

from __future__ import annotations

import pandas as pd

from pearls_aqi.alerts import AlertService, generate_alerts
from pearls_aqi.alerts.base import AlertType
from pearls_aqi.alerts.channels import CollectingAlertChannel
from pearls_aqi.config.models import AlertsConfig


def _forecast(values):
    idx = pd.date_range("2025-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": idx, "predicted_aqi": values})


def test_no_alerts_when_clean():
    cfg = AlertsConfig()
    alerts = generate_alerts(_forecast([40] * 24), cfg, city="Karachi")
    assert alerts == []


def test_all_alert_types_trigger():
    cfg = AlertsConfig()
    # Clean, then a sharp jump to 320 sustained for 3h (spike + sustained +
    # threshold + imminent hazard within 24h).
    values = [80] * 10 + [320, 320, 320] + [90] * 11
    alerts = generate_alerts(_forecast(values), cfg, city="Karachi")
    types = {a.type for a in alerts}
    assert AlertType.THRESHOLD.value in types
    assert AlertType.SUSTAINED.value in types
    assert AlertType.SPIKE.value in types
    assert AlertType.IMMINENT.value in types
    for a in alerts:
        assert a.city == "Karachi"
        assert a.peak_aqi >= cfg.unhealthy_threshold


def test_sustained_requires_consecutive_hours():
    cfg = AlertsConfig(consecutive_hours=3)
    # Only 2 consecutive exceedances -> no SUSTAINED alert.
    values = [80] * 10 + [160, 160] + [80] * 12
    types = {a.type for a in generate_alerts(_forecast(values), cfg)}
    assert AlertType.SUSTAINED.value not in types
    assert AlertType.THRESHOLD.value in types


def test_alert_service_dedup_and_collect():
    cfg = AlertsConfig()
    values = [80] * 10 + [320, 320, 320] + [90] * 11
    alerts = generate_alerts(_forecast(values), cfg, city="Karachi")
    channel = CollectingAlertChannel()
    service = AlertService(channels=[channel])

    first = service.dispatch(alerts)
    assert len(first) == len(alerts)
    assert len(channel.alerts) == len(alerts)

    # Re-dispatching the same events is suppressed.
    second = service.dispatch(alerts)
    assert second == []
