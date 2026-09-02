#!/usr/bin/env python
"""Seed deterministic SAMPLE data into the local feature store.

Runs the real feature pipeline with the sample providers, so the stored rows are
genuine engineered features (only the *source data* is synthetic, tagged
``data_source='sample'``). Enables the whole project to run offline.

    python scripts/seed_sample_data.py --city Karachi --days 180
"""

from __future__ import annotations

import argparse
from datetime import timedelta

from _common import bootstrap, logger, resolve_location
from pearls_aqi.api_clients.sample_provider import (
    SampleAirQualityProvider,
    SampleWeatherProvider,
)
from pearls_aqi.pipelines import FeaturePipeline
from pearls_aqi.storage import get_feature_store
from pearls_aqi.utils.timeutils import now_utc, utc_floor_hour


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed sample data into the local feature store.")
    parser.add_argument("--city", default=None, help="City name (defaults to config).")
    parser.add_argument("--days", type=int, default=180, help="Days of history to generate.")
    args = parser.parse_args()

    config = bootstrap()
    location = resolve_location(config, args.city)
    fs = get_feature_store(config)

    pipeline = FeaturePipeline(
        config,
        aq_provider=SampleAirQualityProvider(),
        weather_provider=SampleWeatherProvider(),
        feature_store=fs,
    )
    end = utc_floor_hour(now_utc())
    start = end - timedelta(days=args.days)
    logger.info("Seeding %d days of sample data for %s ...", args.days, location.city)
    summary = pipeline.run(location, start, end)
    logger.info(
        "Seed complete: %d rows written to '%s' feature store.",
        summary.written,
        config.feature_store.backend,
    )
    print(
        f"Seeded {summary.written} rows for {location.city} "
        f"({args.days} days). You can now run training: make train"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
