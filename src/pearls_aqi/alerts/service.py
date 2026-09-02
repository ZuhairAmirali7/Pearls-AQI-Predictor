"""Alert dispatch service with de-duplication."""

from __future__ import annotations

from collections.abc import Callable

from pearls_aqi.alerts.base import AlertChannel, AQIAlert
from pearls_aqi.alerts.channels import (
    CollectingAlertChannel,
    EmailAlertChannel,
    LogAlertChannel,
    WebhookAlertChannel,
)
from pearls_aqi.config.models import AlertsConfig
from pearls_aqi.utils.logging import get_logger

logger = get_logger(__name__)

_CHANNEL_BUILDERS: dict[str, Callable[[], AlertChannel]] = {
    "log": LogAlertChannel,
    "dashboard": CollectingAlertChannel,
    "email": EmailAlertChannel,
    "webhook": WebhookAlertChannel,
}


def build_channels(cfg: AlertsConfig) -> list[AlertChannel]:
    """Instantiate the configured channels (unknown names are skipped)."""
    channels: list[AlertChannel] = []
    for name in cfg.channels:
        builder = _CHANNEL_BUILDERS.get(name)
        if builder is None:
            logger.warning("Unknown alert channel '%s' — skipping.", name)
            continue
        channels.append(builder())
    return channels


class AlertService:
    """Dispatches alerts to channels, suppressing duplicate events."""

    def __init__(self, channels: list[AlertChannel] | None = None) -> None:
        self.channels = channels or [LogAlertChannel()]
        self._seen: set[str] = set()

    def dispatch(self, alerts: list[AQIAlert]) -> list[AQIAlert]:
        """Send new alerts; return the ones actually dispatched (deduped)."""
        dispatched: list[AQIAlert] = []
        for alert in alerts:
            if alert.event_key in self._seen:
                continue
            self._seen.add(alert.event_key)
            for channel in self.channels:
                try:
                    channel.send(alert)
                except Exception as exc:
                    logger.error("Channel %s failed: %s", getattr(channel, "name", "?"), exc)
            dispatched.append(alert)
        return dispatched
