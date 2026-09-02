"""Page 2 — Current pollutant concentrations and health interpretation."""

from __future__ import annotations

import app._bootstrap  # noqa: F401
import pandas as pd
import streamlit as st
from app.components.charts import pollutant_bar
from app.services import get_config, get_current, get_history, list_cities

from pearls_aqi.aqi import pollutant_subindex
from pearls_aqi.data.domain import categorize_aqi

st.set_page_config(page_title="Current Pollutants", page_icon="🧪", layout="wide")

_LABELS = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "o3": "O₃",
    "no2": "NO₂",
    "so2": "SO₂",
    "co": "CO",
    "nh3": "NH₃",
}


@st.cache_data(ttl=600, show_spinner=False)
def _current(city: str) -> dict:
    return get_current(city)


@st.cache_data(ttl=600, show_spinner=False)
def _history(city: str) -> pd.DataFrame:
    return get_history(city)


def main() -> None:
    cfg = get_config()
    cities = list_cities()
    city = st.sidebar.selectbox("City", cities, index=cities.index(cfg.location.city))

    st.title("🧪 Current Pollutants")
    current = _current(city)
    if not current.get("available"):
        st.error("No current data: " + str(current.get("reason", "")))
        st.stop()

    pollutants = current.get("pollutants", {})
    st.plotly_chart(pollutant_bar(pollutants), use_container_width=True)

    st.subheader("Pollutant detail (with sub-index)")
    rows = []
    for key, label in _LABELS.items():
        conc = pollutants.get(key)
        sub = (
            pollutant_subindex(key, conc)
            if (key in {"pm2_5", "pm10", "o3", "no2", "so2", "co"} and conc is not None)
            else None
        )
        cat = categorize_aqi(sub).value if sub is not None else "—"
        rows.append(
            {
                "Pollutant": label,
                "µg/m³": None if conc is None else round(conc, 1),
                "Sub-index (AQI)": sub,
                "Category": cat,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Sub-indices use the US EPA breakpoints. The overall AQI equals the highest "
        "pollutant sub-index (the *dominant pollutant*)."
    )

    # Recent trends.
    hist = _history(city)
    if hist is not None and not hist.empty:
        st.subheader("Recent pollutant trends (last 7 days)")
        h = hist.copy()
        h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
        recent = h[h["timestamp"] >= h["timestamp"].max() - pd.Timedelta(days=7)]
        cols = [c for c in ("pm2_5", "pm10", "o3", "no2") if c in recent.columns]
        if cols:
            st.line_chart(recent.set_index("timestamp")[cols])


main()
