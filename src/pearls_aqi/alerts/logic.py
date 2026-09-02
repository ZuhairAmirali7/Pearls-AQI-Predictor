"""Hazard-alert detection from a 72-hour forecast.

Rules (all thresholds configurable):
  1. THRESHOLD  — any hour's AQI >= unhealthy threshold.
  2. SUSTAINED  — >= N consecutive hours above threshold.
  3. SPIKE      — AQI rises by >= spike_delta within spike_window_hours.
  4. IMMINENT   — hazardous conditions predicted within the next 24 hours.
"""

from __future__ import annotations

import pandas as pd

from pearls_aqi.alerts.base import AlertSeverity, AlertType, AQIAlert
from pearls_aqi.config.models import AlertsConfig


def _severity_for(aqi: float, cfg: AlertsConfig) -> str:
    if aqi >= cfg.hazardous_threshold:
        return AlertSeverity.HAZARDOUS.value
    if aqi >= cfg.very_unhealthy_threshold:
        return AlertSeverity.VERY_UNHEALTHY.value
    if aqi >= cfg.unhealthy_threshold:
        return AlertSeverity.UNHEALTHY.value
    return AlertSeverity.UNHEALTHY_SENSITIVE.value


def generate_alerts(
    forecast: pd.DataFrame,
    cfg: AlertsConfig,
    city: str | None = None,
    aqi_column: str = "predicted_aqi",
    time_column: str = "timestamp",
) -> list[AQIAlert]:
    """Return a list of :class:`AQIAlert` for a forecast frame.

    ``forecast`` must be hourly and sorted, with ``timestamp`` and
    ``predicted_aqi`` columns.
    """
    if forecast.empty or aqi_column not in forecast.columns:
        return []
    df = forecast.sort_values(time_column).reset_index(drop=True)
    ts = pd.to_datetime(df[time_column], utc=True)
    aqi = pd.to_numeric(df[aqi_column], errors="coerce")
    alerts: list[AQIAlert] = []
    threshold = cfg.unhealthy_threshold

    over = aqi >= threshold
    if over.any():
        peak_idx = int(aqi.idxmax())
        alerts.append(
            AQIAlert(
                type=AlertType.THRESHOLD.value,
                severity=_severity_for(float(aqi.max()), cfg),
                message=(
                    f"AQI is forecast to reach {aqi.max():.0f} "
                    f"(at {ts[peak_idx]:%Y-%m-%d %H:%M UTC}), exceeding the "
                    f"unhealthy threshold of {threshold:.0f}."
                ),
                peak_aqi=float(aqi.max()),
                start_time=ts[over.idxmax()].to_pydatetime(),
                end_time=ts[peak_idx].to_pydatetime(),
                city=city,
            )
        )

    # Sustained: longest run of consecutive exceedances.
    run_start: int | None = None
    best_run: tuple[int, int, int] = (0, 0, 0)
    length = 0
    for i, flag in enumerate(over):
        if flag:
            length = length + 1 if run_start is not None else 1
            run_start = run_start if run_start is not None else i
            if length > best_run[0]:
                best_run = (length, run_start, i)
        else:
            run_start, length = None, 0
    if best_run[0] >= cfg.consecutive_hours:
        s, e = best_run[1], best_run[2]
        alerts.append(
            AQIAlert(
                type=AlertType.SUSTAINED.value,
                severity=_severity_for(float(aqi[s : e + 1].max()), cfg),
                message=(
                    f"AQI is forecast to stay above {threshold:.0f} for "
                    f"{best_run[0]} consecutive hours "
                    f"({ts[s]:%m-%d %H:%M}–{ts[e]:%m-%d %H:%M} UTC)."
                ),
                peak_aqi=float(aqi[s : e + 1].max()),
                start_time=ts[s].to_pydatetime(),
                end_time=ts[e].to_pydatetime(),
                city=city,
            )
        )

    # Spike: max rise within the rolling window.
    win = max(1, cfg.spike_window_hours)
    for i in range(len(aqi)):
        j = min(len(aqi) - 1, i + win)
        rise = aqi[i : j + 1].max() - aqi[i]
        if rise >= cfg.spike_delta:
            peak_j = int(aqi[i : j + 1].idxmax())
            alerts.append(
                AQIAlert(
                    type=AlertType.SPIKE.value,
                    severity=_severity_for(float(aqi[peak_j]), cfg),
                    message=(
                        f"Rapid AQI increase of {rise:.0f} points forecast within "
                        f"{win} hours starting {ts[i]:%m-%d %H:%M UTC}."
                    ),
                    peak_aqi=float(aqi[peak_j]),
                    start_time=ts[i].to_pydatetime(),
                    end_time=ts[peak_j].to_pydatetime(),
                    city=city,
                )
            )
            break  # report only the first spike to avoid noise

    # Imminent hazard within 24h.
    next24 = df[ts <= ts.iloc[0] + pd.Timedelta(hours=24)]
    if not next24.empty:
        a24 = pd.to_numeric(next24[aqi_column], errors="coerce")
        if (a24 >= cfg.hazardous_threshold).any():
            alerts.append(
                AQIAlert(
                    type=AlertType.IMMINENT.value,
                    severity=AlertSeverity.HAZARDOUS.value,
                    message=(
                        f"Hazardous AQI (>= {cfg.hazardous_threshold:.0f}) predicted "
                        "within the next 24 hours. Avoid outdoor exertion."
                    ),
                    peak_aqi=float(a24.max()),
                    start_time=ts.iloc[0].to_pydatetime(),
                    end_time=ts.iloc[0].to_pydatetime() + pd.Timedelta(hours=24),
                    city=city,
                )
            )
    return alerts
