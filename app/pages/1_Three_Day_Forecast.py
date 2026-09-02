"""Page 1 — Three-Day (72h) hourly AQI forecast."""

from __future__ import annotations

import app._bootstrap  # noqa: F401
import pandas as pd
import streamlit as st
from app.components.aqi import category_badge
from app.components.charts import forecast_chart
from app.services import get_config, get_forecast, list_cities

st.set_page_config(page_title="Three-Day Forecast", page_icon="📈", layout="wide")


@st.cache_data(ttl=600, show_spinner="Generating forecast…")
def _forecast(city: str) -> dict:
    return get_forecast(city)


def main() -> None:
    cfg = get_config()
    cities = list_cities()
    city = st.sidebar.selectbox("City", cities, index=cities.index(cfg.location.city))
    tz = st.sidebar.selectbox("Display timezone", sorted({cfg.display.timezone, "UTC"}))

    st.title("📈 Three-Day AQI Forecast")
    data = _forecast(city)
    if not data.get("available"):
        st.error(
            "Forecast unavailable: "
            + str(data.get("reason", ""))
            + "\n\nTry `make seed && make train`."
        )
        st.stop()

    fc = data["forecast"].copy()
    fc["timestamp"] = pd.to_datetime(fc["timestamp"], utc=True)
    meta = data["meta"]
    st.caption(
        f"Model **{meta.get('model_name')}** v{meta.get('model_version')} · "
        f"generated {meta.get('generated_at')} (UTC)"
    )

    st.plotly_chart(forecast_chart(fc, tz), use_container_width=True)

    # Daily summary cards.
    st.subheader("Daily summary")
    fc_local = fc.assign(day=fc["timestamp"].dt.tz_convert(tz).dt.date)
    daily = fc_local.groupby("day")["predicted_aqi"].agg(["min", "mean", "max"]).reset_index()
    cols = st.columns(len(daily))
    for col, (_, r) in zip(cols, daily.iterrows(), strict=False):
        col.metric(f"{r['day']}", f"{r['mean']:.0f} avg")
        col.markdown(category_badge(r["max"]), unsafe_allow_html=True)
        col.caption(f"min {r['min']:.0f} · max {r['max']:.0f}")

    # Highest-risk period.
    peak = fc.loc[fc["predicted_aqi"].idxmax()]
    st.warning(
        f"⚠️ Highest-risk hour: **{peak['predicted_aqi']:.0f} AQI** "
        f"({peak['category']}) at {peak['timestamp'].tz_convert(tz):%Y-%m-%d %H:%M} ({tz})."
    )

    # Forecast table + download.
    st.subheader("Hourly forecast table")
    show = fc.copy()
    show["timestamp"] = show["timestamp"].dt.tz_convert(tz)
    st.dataframe(show, use_container_width=True, height=300)
    st.download_button(
        "⬇️ Download forecast (CSV)",
        data=fc.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{city}.csv",
        mime="text/csv",
    )


main()
