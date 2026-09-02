"""Unit tests for alert channels and channel construction."""

from __future__ import annotations

from datetime import UTC, datetime

from pearls_aqi.alerts.base import AQIAlert
from pearls_aqi.alerts.channels import (
    CollectingAlertChannel,
    EmailAlertChannel,
    LogAlertChannel,
    WebhookAlertChannel,
)
from pearls_aqi.alerts.service import build_channels
from pearls_aqi.config.models import AlertsConfig


def _alert():
    now = datetime(2025, 1, 1, tzinfo=UTC)
    return AQIAlert(
        type="threshold_exceeded",
        severity="unhealthy",
        message="Test alert",
        peak_aqi=180.0,
        start_time=now,
        end_time=now,
        city="Karachi",
    )


def test_log_channel_does_not_raise():
    LogAlertChannel().send(_alert())  # should log without error


def test_collecting_channel_accumulates():
    channel = CollectingAlertChannel()
    channel.send(_alert())
    assert len(channel.alerts) == 1


def test_email_channel_noop_when_unconfigured(monkeypatch):
    for var in ("SMTP_HOST", "ALERT_EMAIL_TO"):
        monkeypatch.delenv(var, raising=False)
    # Not configured -> returns without attempting to send.
    EmailAlertChannel().send(_alert())


def test_webhook_channel_noop_when_unconfigured(monkeypatch):
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    WebhookAlertChannel().send(_alert())


def test_build_channels_from_config():
    cfg = AlertsConfig(channels=["log", "dashboard", "email", "webhook"])
    channels = build_channels(cfg)
    assert len(channels) == 4
    names = {getattr(c, "name", None) for c in channels}
    assert {"log", "dashboard", "email", "webhook"} <= names


def test_build_channels_skips_unknown():
    cfg = AlertsConfig(channels=["log", "carrier_pigeon"])
    channels = build_channels(cfg)
    assert len(channels) == 1


def test_alert_event_key_is_deterministic():
    a1, a2 = _alert(), _alert()
    assert a1.event_key == a2.event_key
