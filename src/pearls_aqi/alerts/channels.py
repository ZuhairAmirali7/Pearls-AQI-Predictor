"""Alert delivery channels (pluggable). None send externally by default."""

from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText

import requests

from pearls_aqi.alerts.base import AQIAlert
from pearls_aqi.utils.logging import get_logger

logger = get_logger(__name__)


class LogAlertChannel:
    """Writes alerts to the application log (safe default)."""

    name = "log"

    def send(self, alert: AQIAlert) -> None:
        logger.warning("[AQI ALERT] %s | %s", alert.severity.upper(), alert.message)


class CollectingAlertChannel:
    """In-memory sink used by the dashboard/API to render banners."""

    name = "dashboard"

    def __init__(self) -> None:
        self.alerts: list[AQIAlert] = []

    def send(self, alert: AQIAlert) -> None:
        self.alerts.append(alert)


class EmailAlertChannel:
    """Sends alerts by SMTP. Requires SMTP_* env vars; opt-in only."""

    name = "email"

    def __init__(self) -> None:
        self.host = os.getenv("SMTP_HOST")
        self.port = int(os.getenv("SMTP_PORT") or 587)
        self.username = os.getenv("SMTP_USERNAME")
        self.password = os.getenv("SMTP_PASSWORD")
        self.to = os.getenv("ALERT_EMAIL_TO")

    def send(self, alert: AQIAlert) -> None:
        if not (self.host and self.to):
            logger.info("Email channel not configured; skipping alert email.")
            return
        msg = MIMEText(alert.message)
        msg["Subject"] = f"AQI Alert ({alert.severity}) — {alert.city or 'city'}"
        msg["From"] = self.username or "aqi-alerts@pearls"
        msg["To"] = self.to
        try:
            with smtplib.SMTP(self.host, self.port, timeout=15) as server:
                server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.sendmail(msg["From"], [self.to], msg.as_string())
            logger.info("Sent alert email to %s", self.to)
        except Exception as exc:
            logger.error("Failed to send alert email: %s", exc)


class WebhookAlertChannel:
    """POSTs the alert JSON to ALERT_WEBHOOK_URL (e.g. Slack). Opt-in only."""

    name = "webhook"

    def __init__(self) -> None:
        self.url = os.getenv("ALERT_WEBHOOK_URL")

    def send(self, alert: AQIAlert) -> None:
        if not self.url:
            logger.info("Webhook channel not configured; skipping.")
            return
        try:
            requests.post(self.url, json={"text": alert.message, **alert.as_dict()}, timeout=10)
        except Exception as exc:
            logger.error("Failed to POST alert webhook: %s", exc)
