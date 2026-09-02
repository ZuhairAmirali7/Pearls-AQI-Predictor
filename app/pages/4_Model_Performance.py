"""Page 4 — Model performance, comparison, and error-by-horizon."""

from __future__ import annotations

import app._bootstrap  # noqa: F401
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app.services import (
    get_config,
    get_model_info,
    get_model_performance,
    get_training_report,
    list_cities,
)

st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")


def main() -> None:
    cfg = get_config()
    cities = list_cities()
    city = st.sidebar.selectbox("City", cities, index=cities.index(cfg.location.city))

    st.title("📊 Model Performance")
    info = get_model_info(city)
    if not info.get("available"):
        st.error("No approved model: " + str(info.get("reason", "")) + "\n\nRun `make train`.")
        st.stop()

    metrics = info.get("metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", f"{info.get('model_type')}")
    c2.metric("Version", info.get("version"))
    c3.metric("Validation MAE", f"{metrics.get('validation_mae', float('nan')):.2f}")
    c4.metric("Validation R²", f"{metrics.get('validation_r2', float('nan')):.3f}")
    c1.metric("Test MAE", f"{metrics.get('test_mae', float('nan')):.2f}")
    c2.metric("Test RMSE", f"{metrics.get('test_rmse', float('nan')):.2f}")
    c3.metric("Hazard MAE", f"{metrics.get('hazard_mae', float('nan')):.2f}")
    c4.metric("Trained", str(info.get("created_at", ""))[:10])
    st.caption(
        "Metrics shown are from the training run's chronological validation/test splits. "
        "The test split is used for reporting only, never for model selection."
    )

    # Model comparison table from the training report.
    report = get_training_report(city)
    if report:
        st.subheader("Model comparison (validation)")
        rows = []
        for r in report.get("results", []):
            if r.get("error"):
                rows.append({"model": r["name"], "note": r["error"]})
                continue
            rows.append(
                {
                    "model": r["name"],
                    "val_MAE": round(r["validation"].get("mae", float("nan")), 2),
                    "val_RMSE": round(r["validation"].get("rmse", float("nan")), 2),
                    "val_R2": round(r["validation"].get("r2", float("nan")), 3),
                    "test_MAE": round(r.get("test", {}).get("mae", float("nan")), 2),
                    "hazard_MAE": round(r.get("hazard", {}).get("hazard_mae", float("nan")), 2),
                }
            )
        st.dataframe(
            pd.DataFrame(rows).sort_values("val_MAE"), use_container_width=True, hide_index=True
        )
        st.caption(
            f"Selected: **{report.get('selected')}** · promotion: {report.get('promotion', {}).get('promote')}"
        )

    # Error by horizon.
    by_h = metrics.get("by_horizon_mae")
    if by_h:
        st.subheader("Error by forecast horizon")
        fig = go.Figure(go.Scatter(x=list(range(1, len(by_h) + 1)), y=by_h, mode="lines+markers"))
        fig.update_layout(
            xaxis_title="Horizon (hours ahead)", yaxis_title="Validation MAE", height=340
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Forecast skill typically degrades as the horizon grows.")

    # Live monitoring (actual vs predicted), if prediction history exists.
    perf = get_model_performance(city)
    if perf.get("available"):
        st.subheader("Live monitoring (observed vs earlier predictions)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Rolling MAE", f"{perf['rolling_mae']:.2f}")
        m2.metric("Rolling RMSE", f"{perf['rolling_rmse']:.2f}")
        m3.metric("Bias", f"{perf['bias']:+.2f}")
        if perf.get("error_by_category"):
            st.dataframe(
                pd.DataFrame(perf["error_by_category"]), use_container_width=True, hide_index=True
            )
    else:
        st.info("Live monitoring will populate once forecasts and later observations overlap.")


main()
