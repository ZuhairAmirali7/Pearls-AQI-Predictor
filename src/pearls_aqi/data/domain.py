"""Core domain models shared across the whole system.

Kept dependency-free (only pydantic + stdlib) so every layer — providers,
features, storage, API — can import it without pulling in heavy modules.
"""

from __future__ import annotations

import re
import unicodedata
from enum import Enum

from pydantic import BaseModel, Field


class Location(BaseModel):
    """A geographic location the system can forecast for."""

    model_config = {"frozen": True}

    city: str
    country: str
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    timezone: str

    @property
    def city_id(self) -> str:
        """Stable, slug-style identifier used in primary keys and paths."""
        return city_id(self.city, self.country)


def city_id(city: str, country: str | None = None) -> str:
    """Deterministic slug for a city (optionally namespaced by country).

    ``city_id("Karachi", "Pakistan") -> "karachi_pakistan"``. Used as part of
    the ``city_id + timestamp`` primary key so records never collide or
    duplicate across ingestion runs.
    """
    parts = [city] if country is None else [city, country]
    slug_parts = []
    for part in parts:
        normalized = unicodedata.normalize("NFKD", part)
        ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
        slug = re.sub(r"[^a-z0-9]+", "_", ascii_only.lower()).strip("_")
        if slug:
            slug_parts.append(slug)
    return "_".join(slug_parts)


class AQICategory(str, Enum):
    """US EPA AQI categories (name + colour used consistently in the UI)."""

    GOOD = "Good"
    MODERATE = "Moderate"
    UNHEALTHY_SENSITIVE = "Unhealthy for Sensitive Groups"
    UNHEALTHY = "Unhealthy"
    VERY_UNHEALTHY = "Very Unhealthy"
    HAZARDOUS = "Hazardous"

    @property
    def color(self) -> str:
        """Official EPA hex colour for this category."""
        return _CATEGORY_COLORS[self]

    @property
    def icon(self) -> str:
        """Emoji/text icon — colour is never the sole hazard signal."""
        return _CATEGORY_ICONS[self]

    @property
    def health_message(self) -> str:
        return _CATEGORY_HEALTH[self]


_CATEGORY_COLORS: dict[AQICategory, str] = {
    AQICategory.GOOD: "#00E400",
    AQICategory.MODERATE: "#FFFF00",
    AQICategory.UNHEALTHY_SENSITIVE: "#FF7E00",
    AQICategory.UNHEALTHY: "#FF0000",
    AQICategory.VERY_UNHEALTHY: "#8F3F97",
    AQICategory.HAZARDOUS: "#7E0023",
}

_CATEGORY_ICONS: dict[AQICategory, str] = {
    AQICategory.GOOD: "🟢",
    AQICategory.MODERATE: "🟡",
    AQICategory.UNHEALTHY_SENSITIVE: "🟠",
    AQICategory.UNHEALTHY: "🔴",
    AQICategory.VERY_UNHEALTHY: "🟣",
    AQICategory.HAZARDOUS: "🟤",
}

_CATEGORY_HEALTH: dict[AQICategory, str] = {
    AQICategory.GOOD: "Air quality is satisfactory and poses little or no risk.",
    AQICategory.MODERATE: (
        "Air quality is acceptable; unusually sensitive people should consider "
        "limiting prolonged outdoor exertion."
    ),
    AQICategory.UNHEALTHY_SENSITIVE: (
        "Members of sensitive groups may experience health effects; the general "
        "public is less likely to be affected."
    ),
    AQICategory.UNHEALTHY: (
        "Some members of the general public may experience health effects; "
        "sensitive groups may experience more serious effects."
    ),
    AQICategory.VERY_UNHEALTHY: (
        "Health alert: the risk of health effects is increased for everyone."
    ),
    AQICategory.HAZARDOUS: (
        "Health warning of emergency conditions; everyone is more likely to be "
        "affected. Avoid outdoor exertion."
    ),
}

# Upper inclusive bound -> category, ordered. AQI is defined on [0, 500].
_CATEGORY_BREAKPOINTS: list[tuple[float, AQICategory]] = [
    (50, AQICategory.GOOD),
    (100, AQICategory.MODERATE),
    (150, AQICategory.UNHEALTHY_SENSITIVE),
    (200, AQICategory.UNHEALTHY),
    (300, AQICategory.VERY_UNHEALTHY),
    (500, AQICategory.HAZARDOUS),
]


def categorize_aqi(aqi: float) -> AQICategory:
    """Map a numeric AQI value to its US EPA category.

    Values above 500 are clamped to ``HAZARDOUS`` (the top defined category);
    negatives are treated as ``Good``. Raises ``ValueError`` for NaN/None.
    """
    if aqi is None or aqi != aqi:  # NaN check
        raise ValueError("AQI value must be a finite number, got NaN/None.")
    for upper, category in _CATEGORY_BREAKPOINTS:
        if aqi <= upper:
            return category
    return AQICategory.HAZARDOUS
