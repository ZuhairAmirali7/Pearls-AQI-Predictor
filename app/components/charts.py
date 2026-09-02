"""Plotly chart builders for the dashboard (theme-agnostic)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from app.components.aqi import CATEGORY_BANDS


def _add_category_bands(fig: go.Figure, x0, x1, y_max: float = 320) -> None:
    for lo, hi, color, _label in CATEGORY_BANDS:
        if lo > y_max:
            break
        fig.add_hrect(
            y0=lo, y1=min(hi, y_max), fillcolor=color, opacity=0.10, line_width=0, layer="below"
        )


def forecast_chart(forecast: pd.DataFrame, tz_name: str = "UTC") -> go.Figure:
    """Hourly AQI forecast with prediction interval + category background bands."""
    df = forecast.copy()
    x = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(tz_name)
    y_max = max(320, float(df["upper_bound"].max()) + 20 if "upper_bound" in df else 320)
    fig = go.Figure()
    _add_category_bands(fig, x.min(), x.max(), y_max)
    if {"lower_bound", "upper_bound"} <= set(df.columns):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["upper_bound"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["lower_bound"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(100,100,100,0.20)",
                name="~80% interval",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["predicted_aqi"],
            mode="lines+markers",
            name="Predicted AQI",
            line={"color": "#1f77b4", "width": 2},
            marker={"size": 4},
        )
    )
    fig.update_layout(
        title="Hourly AQI forecast (next 72 hours)",
        xaxis_title=f"Time ({tz_name})",
        yaxis_title="AQI (US EPA)",
        yaxis={"range": [0, y_max]},
        hovermode="x unified",
        height=430,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def history_chart(history: pd.DataFrame, tz_name: str = "UTC") -> go.Figure:
    df = history.copy()
    x = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(tz_name)
    fig = go.Figure()
    _add_category_bands(fig, x.min(), x.max())
    fig.add_trace(
        go.Scatter(
            x=x, y=df["aqi"], mode="lines", name="Observed AQI", line={"color": "#444", "width": 1}
        )
    )
    fig.update_layout(
        title="Historical AQI",
        xaxis_title=f"Time ({tz_name})",
        yaxis_title="AQI",
        height=400,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def pattern_chart(series: pd.Series, x_label: str, title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=series.index.astype(str), y=series.values, marker_color="#1f77b4"))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Mean AQI",
        height=320,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return fig


def pollutant_bar(pollutants: dict[str, float]) -> go.Figure:
    labels = {
        "pm2_5": "PM2.5",
        "pm10": "PM10",
        "o3": "O₃",
        "no2": "NO₂",
        "so2": "SO₂",
        "co": "CO",
        "nh3": "NH₃",
    }
    items = [(labels.get(k, k), v) for k, v in pollutants.items() if v is not None]
    fig = go.Figure(
        go.Bar(x=[i[0] for i in items], y=[i[1] for i in items], marker_color="#ff7e00")
    )
    fig.update_layout(
        title="Current pollutant concentrations (µg/m³)",
        yaxis_title="µg/m³",
        height=320,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return fig


def actual_vs_predicted(pairs: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pairs["actual"],
            y=pairs["predicted"],
            mode="markers",
            marker={"size": 5, "opacity": 0.5},
            name="obs",
        )
    )
    lo = float(min(pairs["actual"].min(), pairs["predicted"].min()))
    hi = float(max(pairs["actual"].max(), pairs["predicted"].max()))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="ideal",
            line={"color": "red", "dash": "dash"},
        )
    )
    fig.update_layout(
        title="Actual vs Predicted AQI",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        height=380,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return fig


def importance_bar(importance: pd.DataFrame, value_col: str = "importance") -> go.Figure:
    df = importance.head(15).iloc[::-1]
    fig = go.Figure(
        go.Bar(x=df[value_col], y=df["feature"], orientation="h", marker_color="#2ca02c")
    )
    fig.update_layout(
        title="Feature importance",
        xaxis_title=value_col,
        height=440,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return fig
