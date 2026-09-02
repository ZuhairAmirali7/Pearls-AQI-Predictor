"""Hazard alerts: detection logic, pluggable channels, and dispatch."""

from __future__ import annotations

from pearls_aqi.alerts.base import AlertChannel, AlertType, AQIAlert
from pearls_aqi.alerts.logic import generate_alerts
from pearls_aqi.alerts.service import AlertService, build_channels

__all__ = [
    "AQIAlert",
    "AlertChannel",
    "AlertService",
    "AlertType",
    "build_channels",
    "generate_alerts",
]
