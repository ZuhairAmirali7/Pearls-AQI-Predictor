"""Alert domain types and the channel interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class AlertType(str, Enum):
    THRESHOLD = "threshold_exceeded"
    SUSTAINED = "sustained_exceedance"
    SPIKE = "rapid_increase"
    IMMINENT = "hazard_within_24h"


class AlertSeverity(str, Enum):
    UNHEALTHY_SENSITIVE = "unhealthy_sensitive"
    UNHEALTHY = "unhealthy"
    VERY_UNHEALTHY = "very_unhealthy"
    HAZARDOUS = "hazardous"


@dataclass
class AQIAlert:
    """A single hazard alert derived from a forecast."""

    type: str
    severity: str
    message: str
    peak_aqi: float
    start_time: datetime
    end_time: datetime
    city: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def event_key(self) -> str:
        """Deterministic key for de-duplication (same event => same key)."""
        return f"{self.city}|{self.type}|{self.severity}|{self.start_time.isoformat()}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "peak_aqi": self.peak_aqi,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "city": self.city,
            "metadata": self.metadata,
        }


@runtime_checkable
class AlertChannel(Protocol):
    """A destination an alert can be delivered to."""

    name: str

    def send(self, alert: AQIAlert) -> None: ...
