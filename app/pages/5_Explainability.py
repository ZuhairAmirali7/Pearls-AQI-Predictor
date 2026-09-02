"""Page 5 — Forecast explainability (feature importance + plain language)."""

from __future__ import annotations

import app._bootstrap  # noqa: F401
import pandas as pd
import streamlit as st
from app.components.charts import importance_bar
from app.services import get_config, get_explanation, get_forecast, list_cities

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")


def main() -> None:
    cfg = get_config()
    cities = list_cities()
    city = st.sidebar.selectbox("City", cities, index=cities.index(cfg.location.city))

    st.title("🔍 Explainability")
    st.caption(
        "Feature importance reflects **how the model behaves**, not proven cause and "
        "effect. Method is labelled below."
    )

    # Pick the horizon to explain (default: highest predicted AQI hour).
    forecast = get_forecast(city)
    default_h = 0
    if forecast.get("available"):
        fc = forecast["forecast"]
        default_h = int(fc["predicted_aqi"].idxmax()) if not fc.empty else 0

    horizon_index = st.slider(
        "Forecast horizon to explain (hours ahead − 1)",
        0,
        max(0, cfg.forecast.horizon_hours - 1),
        default_h,
    )

    if not st.button("Compute explanation", type="primary"):
        st.info(
            "Click **Compute explanation** — this runs permutation/occlusion analysis (a few seconds)."
        )
        st.stop()

    with st.spinner("Computing feature importance…"):
        result = get_explanation(city, horizon_index=horizon_index)

    if not result.get("available"):
        st.error("Explanation unavailable: " + str(result.get("reason", "")))
        st.stop()

    st.success(f"Method: **{result.get('method')}** — {result.get('notes')}")

    if result.get("plain_language"):
        st.subheader("In plain language")
        st.write(result["plain_language"])

    gi = pd.DataFrame(result.get("global_importance", []))
    if not gi.empty:
        st.subheader("Global feature importance")
        st.plotly_chart(importance_bar(gi, value_col="importance"), use_container_width=True)

    li = result.get("local_importance")
    if li:
        st.subheader("Local explanation (this forecast point)")
        ldf = pd.DataFrame(li)
        st.dataframe(ldf, use_container_width=True, hide_index=True)
        st.caption("Positive contribution = the feature pushed the predicted AQI up.")


main()
