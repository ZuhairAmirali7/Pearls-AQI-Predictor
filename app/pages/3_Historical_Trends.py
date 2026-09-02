"""Page 3 — Historical AQI trends and seasonal/diurnal patterns."""

from __future__ import annotations

from datetime import timedelta

import app._bootstrap  # noqa: F401
import pandas as pd
import streamlit as st
from app.components.charts import history_chart, pattern_chart
from app.services import get_config, get_history, list_cities

from pearls_aqi.utils.timeutils import now_utc

st.set_page_config(page_title="Historical Trends", page_icon="🗓️", layout="wide")

_DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@st.cache_data(ttl=900, show_spinner="Loading history…")
def _history(city: str) -> pd.DataFrame:
    return get_history(city)


def main() -> None:
    cfg = get_config()
    cities = list_cities()
    city = st.sidebar.selectbox("City", cities, index=cities.index(cfg.location.city))
    tz = st.sidebar.selectbox("Display timezone", sorted({cfg.display.timezone, "UTC"}))

    st.title("🗓️ Historical Trends")
    hist = _history(city)
    if hist is None or hist.empty:
        st.error("No historical data. Run `make seed` or a backfill.")
        st.stop()

    hist = hist.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)
    tmin, tmax = hist["timestamp"].min().date(), hist["timestamp"].max().date()
    default_start = max(tmin, (now_utc() - timedelta(days=60)).date())
    start, end = st.slider(
        "Date range", min_value=tmin, max_value=tmax, value=(default_start, tmax)
    )
    mask = (hist["timestamp"].dt.date >= start) & (hist["timestamp"].dt.date <= end)
    view = hist[mask]

    st.plotly_chart(history_chart(view, tz), use_container_width=True)

    local_ts = view["timestamp"].dt.tz_convert(tz)
    c1, c2 = st.columns(2)
    with c1:
        by_hour = view.assign(hour=local_ts.dt.hour).groupby("hour")["aqi"].mean()
        st.plotly_chart(
            pattern_chart(by_hour, "Hour of day", "AQI by hour of day"), use_container_width=True
        )
    with c2:
        by_dow = view.assign(dow=local_ts.dt.dayofweek).groupby("dow")["aqi"].mean()
        by_dow.index = [_DOW[i] for i in by_dow.index]
        st.plotly_chart(
            pattern_chart(by_dow, "Day of week", "AQI by day of week"), use_container_width=True
        )

    c3, c4 = st.columns(2)
    with c3:
        by_month = view.assign(month=local_ts.dt.month).groupby("month")["aqi"].mean()
        st.plotly_chart(pattern_chart(by_month, "Month", "AQI by month"), use_container_width=True)
    with c4:
        daily = view.set_index("timestamp")["aqi"].resample("1D").mean()
        st.line_chart(daily.rename("Daily mean AQI"))

    st.caption(
        f"Weekend vs weekday mean AQI: "
        f"{view.assign(we=local_ts.dt.dayofweek >= 5).groupby('we')['aqi'].mean().round(1).to_dict()}"
    )


main()
