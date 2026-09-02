"""US EPA Air Quality Index conversion and classification.

This is the single source of truth for turning pollutant *concentrations* into an
AQI value, sub-indices, a dominant pollutant, and a category. It is deliberately
provider-independent and heavily unit-tested (``tests/unit/test_aqi.py``).

Concentration inputs are expected in **micrograms per cubic metre (µg/m³)** —
the unit returned by both Open-Meteo and OpenWeather — *except* CO which may be
supplied in µg/m³ as well. Gaseous pollutants are internally converted to the
EPA reporting units (ppb / ppm) assuming standard conditions (25 °C, 1 atm,
molar volume 24.45 L/mol) before the piecewise-linear AQI formula is applied.

AQI standard: **US EPA**. PM2.5 breakpoints use the 2024 EPA revision
(effective 6 May 2024). Reference: https://www.airnow.gov/aqi/aqi-basics/ and
EPA Technical Assistance Document for the Reporting of Daily Air Quality.

The piecewise-linear formula for a pollutant sub-index is:

    I = (I_hi - I_lo) / (C_hi - C_lo) * (C_trunc - C_lo) + I_lo

The overall AQI is the maximum of the available pollutant sub-indices, and the
dominant pollutant is the one producing that maximum.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from pearls_aqi.data.domain import AQICategory, categorize_aqi

# Molar volume of an ideal gas at 25 °C and 1 atm (litres per mole).
_MOLAR_VOLUME = 24.45

# Molecular weights (g/mol) for gaseous pollutants.
_MW = {"o3": 48.00, "no2": 46.01, "so2": 64.07, "co": 28.01}

# Canonical pollutant keys used everywhere in the project.
POLLUTANTS: tuple[str, ...] = ("pm2_5", "pm10", "o3", "no2", "so2", "co")


def _ugm3_to_ppb(mw: float) -> Callable[[float], float]:
    """Return a µg/m³ → ppb converter for a gas of molecular weight ``mw``."""

    def convert(ugm3: float) -> float:
        return ugm3 * _MOLAR_VOLUME / mw

    return convert


def _ugm3_to_ppm(mw: float) -> Callable[[float], float]:
    """Return a µg/m³ → ppm converter for a gas of molecular weight ``mw``."""
    ppb = _ugm3_to_ppb(mw)
    return lambda ugm3: ppb(ugm3) / 1000.0


def _identity(x: float) -> float:
    return x


def _truncate(value: float, decimals: int) -> float:
    """EPA-style truncation (toward zero) to a fixed number of decimals."""
    factor = 10**decimals
    return math.floor(value * factor) / factor


@dataclass(frozen=True)
class PollutantSpec:
    """AQI breakpoint definition for a single pollutant."""

    key: str
    epa_unit: str
    truncate_decimals: int
    to_epa_unit: Callable[[float], float]
    # (C_low, C_high, I_low, I_high) segments in EPA units, ascending.
    breakpoints: tuple[tuple[float, float, float, float], ...]


# Breakpoint tables (EPA units). Sources: EPA AQI Technical Assistance Document.
_SPECS: dict[str, PollutantSpec] = {
    # PM2.5 — 24-hour average, µg/m³. 2024 EPA revision.
    "pm2_5": PollutantSpec(
        key="pm2_5",
        epa_unit="µg/m³",
        truncate_decimals=1,
        to_epa_unit=_identity,
        breakpoints=(
            (0.0, 9.0, 0, 50),
            (9.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 125.4, 151, 200),
            (125.5, 225.4, 201, 300),
            (225.5, 325.4, 301, 500),
        ),
    ),
    # PM10 — 24-hour average, µg/m³.
    "pm10": PollutantSpec(
        key="pm10",
        epa_unit="µg/m³",
        truncate_decimals=0,
        to_epa_unit=_identity,
        breakpoints=(
            (0, 54, 0, 50),
            (55, 154, 51, 100),
            (155, 254, 101, 150),
            (255, 354, 151, 200),
            (355, 424, 201, 300),
            (425, 504, 301, 400),
            (505, 604, 401, 500),
        ),
    ),
    # O3 — 8-hour average, ppm. (1-hour O3 for >0.200 ppm is not modelled here;
    # we cap at the 8-hour table which covers the operationally common range.)
    "o3": PollutantSpec(
        key="o3",
        epa_unit="ppm",
        truncate_decimals=3,
        to_epa_unit=_ugm3_to_ppm(_MW["o3"]),
        breakpoints=(
            (0.000, 0.054, 0, 50),
            (0.055, 0.070, 51, 100),
            (0.071, 0.085, 101, 150),
            (0.086, 0.105, 151, 200),
            (0.106, 0.200, 201, 300),
        ),
    ),
    # NO2 — 1-hour average, ppb.
    "no2": PollutantSpec(
        key="no2",
        epa_unit="ppb",
        truncate_decimals=0,
        to_epa_unit=_ugm3_to_ppb(_MW["no2"]),
        breakpoints=(
            (0, 53, 0, 50),
            (54, 100, 51, 100),
            (101, 360, 101, 150),
            (361, 649, 151, 200),
            (650, 1249, 201, 300),
            (1250, 1649, 301, 400),
            (1650, 2049, 401, 500),
        ),
    ),
    # SO2 — 1-hour average, ppb (24-hour handling for >304 ppb not modelled).
    "so2": PollutantSpec(
        key="so2",
        epa_unit="ppb",
        truncate_decimals=0,
        to_epa_unit=_ugm3_to_ppb(_MW["so2"]),
        breakpoints=(
            (0, 35, 0, 50),
            (36, 75, 51, 100),
            (76, 185, 101, 150),
            (186, 304, 151, 200),
            (305, 604, 201, 300),
            (605, 804, 301, 400),
            (805, 1004, 401, 500),
        ),
    ),
    # CO — 8-hour average, ppm.
    "co": PollutantSpec(
        key="co",
        epa_unit="ppm",
        truncate_decimals=1,
        to_epa_unit=_ugm3_to_ppm(_MW["co"]),
        breakpoints=(
            (0.0, 4.4, 0, 50),
            (4.5, 9.4, 51, 100),
            (9.5, 12.4, 101, 150),
            (12.5, 15.4, 151, 200),
            (15.5, 30.4, 201, 300),
            (30.5, 40.4, 301, 400),
            (40.5, 50.4, 401, 500),
        ),
    ),
}


@dataclass
class AQIResult:
    """Result of an AQI computation from pollutant concentrations."""

    aqi: float | None
    category: AQICategory | None
    dominant_pollutant: str | None
    sub_indices: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "aqi": self.aqi,
            "category": self.category.value if self.category else None,
            "dominant_pollutant": self.dominant_pollutant,
            "sub_indices": self.sub_indices,
        }


def pollutant_subindex(pollutant: str, concentration_ugm3: float | None) -> float | None:
    """Compute the AQI sub-index for a single pollutant concentration (µg/m³).

    Returns ``None`` when the concentration is missing (``None``/NaN). Never
    silently substitutes zero for missing data. Concentrations above the top
    breakpoint are clamped to the maximum AQI of 500 (documented limitation).
    """
    if pollutant not in _SPECS:
        raise ValueError(f"Unsupported pollutant: {pollutant!r}. Known: {list(_SPECS)}")
    if concentration_ugm3 is None or (
        isinstance(concentration_ugm3, float) and math.isnan(concentration_ugm3)
    ):
        return None
    if concentration_ugm3 < 0:
        raise ValueError(f"Negative concentration for {pollutant}: {concentration_ugm3}")

    spec = _SPECS[pollutant]
    conc = spec.to_epa_unit(float(concentration_ugm3))
    conc = _truncate(conc, spec.truncate_decimals)

    lowest_c = spec.breakpoints[0][0]
    highest_c = spec.breakpoints[-1][1]
    if conc <= lowest_c:
        return float(spec.breakpoints[0][2])
    if conc >= highest_c:
        return float(spec.breakpoints[-1][3])  # clamp to top of table

    for c_lo, c_hi, i_lo, i_hi in spec.breakpoints:
        if c_lo <= conc <= c_hi:
            index = (i_hi - i_lo) / (c_hi - c_lo) * (conc - c_lo) + i_lo
            return round(index)
    # Fall into a gap between breakpoint segments (shouldn't happen with EPA
    # tables, but guard anyway): use the nearest lower segment's high value.
    for _c_lo, c_hi, _i_lo, i_hi in reversed(spec.breakpoints):
        if conc > c_hi:
            return float(i_hi)
    return None


def aqi_from_concentrations(
    concentrations: dict[str, float | None],
    standard: str = "us_epa",
) -> AQIResult:
    """Compute overall AQI, sub-indices, category, and dominant pollutant.

    Args:
        concentrations: Mapping of pollutant key -> concentration in µg/m³. Keys
            outside :data:`POLLUTANTS` are ignored. Missing pollutants are simply
            excluded from the maximum (not treated as zero).
        standard: AQI standard identifier. Only ``"us_epa"`` is implemented.

    Returns:
        An :class:`AQIResult`. If no pollutant has a usable concentration, all
        fields are ``None`` / empty (the caller decides how to handle it).
    """
    if standard != "us_epa":
        raise ValueError(f"Only the 'us_epa' standard is implemented, got {standard!r}.")

    sub_indices: dict[str, float] = {}
    for pollutant in POLLUTANTS:
        value = concentrations.get(pollutant)
        sub = pollutant_subindex(pollutant, value)
        if sub is not None:
            sub_indices[pollutant] = float(sub)

    if not sub_indices:
        return AQIResult(aqi=None, category=None, dominant_pollutant=None, sub_indices={})

    dominant = max(sub_indices, key=lambda k: sub_indices[k])
    overall = sub_indices[dominant]
    return AQIResult(
        aqi=overall,
        category=categorize_aqi(overall),
        dominant_pollutant=dominant,
        sub_indices=sub_indices,
    )


def categorize(aqi: float) -> AQICategory:
    """Public alias for :func:`pearls_aqi.data.domain.categorize_aqi`."""
    return categorize_aqi(aqi)


def supported_pollutants() -> tuple[str, ...]:
    return POLLUTANTS
