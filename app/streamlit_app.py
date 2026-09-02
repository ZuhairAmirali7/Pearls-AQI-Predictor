"""Pearls AQI Predictor — Streamlit dashboard (Overview / home page)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import app._bootstrap  # noqa: F401  (adds repo root to sys.path)
from app.components.aqi import category_badge, health_message
from app.services import (
    get_config,
    get_current,
    get_forecast,
    list_cities,
    mode,
)

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="wide")


def sidebar() -> tuple[str, str]:
    st.sidebar.title("🌫️ Pearls AQI")
    cfg = get_config()
    cities = list_cities()
    default_ix = cities.index(cfg.location.city) if cfg.location.city in cities else 0
    city = st.sidebar.selectbox("City", cities, index=default_ix)
    tz_options = sorted(
        {cfg.display.timezone, cfg.location.timezone, "UTC", "Asia/Karachi", "Asia/Kolkata"}
    )
    tz = st.sidebar.selectbox("Display timezone", tz_options, index=0)
    st.sidebar.caption(f"Data mode: **{mode()}**")
    if st.sidebar.button("↻ Refresh data"):
        st.cache_data.clear()
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Forecasts are model estimates for information only — **not** a substitute "
        "for official air-quality warnings or medical advice."
    )
    return city, tz


@st.cache_data(ttl=600, show_spinner=False)
def _forecast(city: str) -> dict:
    return get_forecast(city)


@st.cache_data(ttl=600, show_spinner=False)
def _current(city: str) -> dict:
    return get_current(city)


def render_alerts(alerts: list[dict]) -> None:
    if not alerts:
        st.success("✅ No hazardous AQI alerts in the next 72 hours.")
        return
    for a in alerts:
        sev = a.get("severity", "")
        msg = f"**{a.get('type', 'alert').replace('_', ' ').title()}** — {a.get('message', '')}"
        if sev in ("hazardous", "very_unhealthy"):
            st.error("🚨 " + msg)
        else:
            st.warning("⚠️ " + msg)


def main() -> None:
    city, tz = sidebar()
    st.title("Air Quality Overview")

    current = _current(city)
    forecast = _forecast(city)

    if not current.get("available") and not forecast.get("available"):
        st.error(
            "No data or model available yet. Run:\n\n"
            "```bash\nmake seed   # sample data\nmake train  # train a model\n```"
        )
        st.stop()

    # --- Current conditions ---
    st.subheader("Current conditions")
    if current.get("available"):
        c1, c2, c3, c4 = st.columns(4)
        aqi = current.get("aqi")
        c1.metric("Current AQI", f"{aqi:.0f}" if aqi is not None else "—")
        c1.markdown(category_badge(aqi), unsafe_allow_html=True)
        c2.metric("Dominant pollutant", str(current.get("dominant_pollutant") or "—").upper())
        wx = current.get("weather") or {}
        temp = wx.get("temperature")
        c3.metric("Temperature", f"{temp:.1f} °C" if temp is not None else "—")
        hum = wx.get("humidity")
        c4.metric("Humidity", f"{hum:.0f}%" if hum is not None else "—")
        st.caption(f"ℹ️ {health_message(aqi)}")
        ts = current.get("timestamp")
        st.caption(
            f"Last updated: {pd.to_datetime(ts).tz_convert(tz) if ts else '—'} "
            f"· source: {current.get('data_source', 'n/a')}"
        )
    else:
        st.info("Current conditions unavailable: " + str(current.get("reason", "")))

    # --- Forecast summary + alerts ---
    st.subheader("Next 72 hours")
    if forecast.get("available"):
        fc = forecast["forecast"]
        meta = forecast["meta"]
        fc = fc.copy()
        fc["timestamp"] = pd.to_datetime(fc["timestamp"], utc=True)
        next24 = fc[fc["timestamp"] <= fc["timestamp"].min() + pd.Timedelta(hours=24)]
        m1, m2, m3 = st.columns(3)
        m1.metric("Next 24h max AQI", f"{next24['predicted_aqi'].max():.0f}")
        m2.metric("Next 72h max AQI", f"{fc['predicted_aqi'].max():.0f}")
        m3.metric("Model", f"{meta.get('model_name', '?')} v{meta.get('model_version', '?')}")
        st.caption(f"Forecast generated at {meta.get('generated_at', 'n/a')} (UTC).")
        render_alerts(forecast.get("alerts", []))
        st.markdown("→ See the **Three-Day Forecast** page for the hourly chart and download.")
    else:
        st.info("Forecast unavailable: " + str(forecast.get("reason", "")))


main()
