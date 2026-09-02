#!/usr/bin/env python
"""Hourly feature pipeline: ingest recent data → build features → write store.

    python scripts/run_feature_pipeline.py --city Karachi

Idempotent — safe to run every hour (rows are upserted on (city_id, timestamp)).
Writes a run summary under artifacts/runs/ and exits non-zero on failure.
"""

from __future__ import annotations

import argparse

from _common import bootstrap, logger, resolve_location
from pearls_aqi.exceptions import PearlsError
from pearls_aqi.pipelines import FeaturePipeline
from pearls_aqi.utils.io import write_json
from pearls_aqi.utils.timeutils import now_utc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the hourly feature pipeline once.")
    parser.add_argument("--city", default=None)
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=None,
        help="Window size; defaults to max lag/rolling window + buffer.",
    )
    args = parser.parse_args()

    config = bootstrap()
    location = resolve_location(config, args.city)
    pipeline = FeaturePipeline(config)

    try:
        summary = pipeline.run_current(location, lookback_hours=args.lookback_hours)
    except PearlsError as exc:
        logger.error("Feature pipeline failed: %s", exc)
        return 1

    stamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    write_json(summary.as_dict(), f"artifacts/runs/{location.city_id}/feature_run_{stamp}.json")
    write_json(summary.as_dict(), f"artifacts/runs/{location.city_id}/feature_run_latest.json")
    print(
        f"Feature pipeline OK for {location.city}: "
        f"fetched={summary.fetched} written={summary.written} "
        f"(warnings={summary.validation.get('n_warnings', 0)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
