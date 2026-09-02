"""Pearls AQI Predictor — serverless air-quality forecasting.

Predicts the US EPA Air Quality Index for a configurable city for the next
72 hours using weather and pollution data, a feature store, and a compared set
of forecasting models.

The public surface is intentionally small; import subpackages directly:

    from pearls_aqi.config import load_config
    from pearls_aqi.aqi import aqi_from_concentrations, categorize
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
