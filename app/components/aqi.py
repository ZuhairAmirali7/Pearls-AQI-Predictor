"""AQI presentation helpers — colour scale plus text + icon (never colour alone)."""

from __future__ import annotations

from pearls_aqi.data.domain import AQICategory, categorize_aqi

# EPA category → (background, text colour) for accessible contrast.
_TEXT_ON = {
    AQICategory.GOOD: "#0b3d0b",
    AQICategory.MODERATE: "#5c5300",
    AQICategory.UNHEALTHY_SENSITIVE: "#7a3b00",
    AQICategory.UNHEALTHY: "#ffffff",
    AQICategory.VERY_UNHEALTHY: "#ffffff",
    AQICategory.HAZARDOUS: "#ffffff",
}


def aqi_color(aqi: float | None) -> str:
    """Hex colour for an AQI value (grey if unknown)."""
    if aqi is None or aqi != aqi:  # NaN
        return "#9e9e9e"
    return categorize_aqi(aqi).color


def category_of(aqi: float | None) -> AQICategory | None:
    if aqi is None or aqi != aqi:
        return None
    return categorize_aqi(aqi)


def category_badge(aqi: float | None) -> str:
    """Return an HTML chip conveying category by colour AND icon AND text."""
    cat = category_of(aqi)
    if cat is None:
        return "<span style='padding:4px 10px;border-radius:12px;background:#9e9e9e;color:#fff;'>❔ Unknown</span>"
    fg = _TEXT_ON[cat]
    return (
        f"<span style='padding:4px 12px;border-radius:12px;font-weight:600;"
        f"background:{cat.color};color:{fg};'>{cat.icon} {cat.value}</span>"
    )


def health_message(aqi: float | None) -> str:
    cat = category_of(aqi)
    return cat.health_message if cat else "No data available."


# Category band definitions for chart backgrounds (lo, hi, colour, label).
CATEGORY_BANDS: list[tuple[float, float, str, str]] = [
    (0, 50, "#00E400", "Good"),
    (50, 100, "#FFFF00", "Moderate"),
    (100, 150, "#FF7E00", "USG"),
    (150, 200, "#FF0000", "Unhealthy"),
    (200, 300, "#8F3F97", "Very Unhealthy"),
    (300, 500, "#7E0023", "Hazardous"),
]
