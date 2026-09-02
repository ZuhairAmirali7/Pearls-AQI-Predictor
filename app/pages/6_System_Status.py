"""Page 6 — System status: pipeline runs, data freshness, model, backends."""

from __future__ import annotations

import app._bootstrap  # noqa: F401
import streamlit as st
from app.services import (
    api_base_url,
    get_config,
    get_feature_run,
    get_system_status,
    get_training_report,
    list_cities,
)

st.set_page_config(page_title="System Status", page_icon="🩺", layout="wide")


def main() -> None:
    cfg = get_config()
    cities = list_cities()
    city = st.sidebar.selectbox("City", cities, index=cities.index(cfg.location.city))

    st.title("🩺 System Status")
    status = get_system_status(city)

    data = status.get("data", {})
    model = status.get("model", {})
    c1, c2, c3 = st.columns(3)
    if data.get("available"):
        stale = data.get("is_stale")
        c1.metric(
            "Data age (h)",
            f"{data.get('data_age_hours', '—')}",
            delta="stale" if stale else "fresh",
            delta_color="inverse" if stale else "normal",
        )
        c2.metric("Rows stored", data.get("rows"))
        c3.metric("Missing rows %", data.get("missing_row_pct"))
    else:
        c1.warning("No feature data")

    st.subheader("Model")
    if model.get("version") is not None:
        st.write(
            f"**{model.get('name')}** v{model.get('version')} "
            f"({model.get('type')}) — status: {model.get('status')} — trained {model.get('trained_at')}"
        )
    else:
        st.warning("No approved model: " + str(model.get("reason", "")))

    st.subheader("Pipelines")
    feat_run = get_feature_run(city)
    st.write(
        "Last feature-pipeline run:", feat_run.get("end") if feat_run else "— (run `make features`)"
    )
    report = get_training_report(city)
    st.write("Last training run:", status.get("last_training_run") or "—")
    if report:
        st.caption(
            f"Registered {report.get('model_name')} v{report.get('registered_version')} "
            f"({report.get('registered_status')})"
        )

    st.subheader("Backends")
    b1, b2, b3 = st.columns(3)
    b1.metric("Feature store", status.get("feature_store_backend", "?"))
    b2.metric("Model registry", status.get("model_registry_backend", "?"))
    b3.metric("API", "connected" if api_base_url() else "direct mode")
    if api_base_url():
        st.caption(f"API base URL: {api_base_url()}")


main()
