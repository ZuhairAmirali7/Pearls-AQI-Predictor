#!/usr/bin/env python
"""Historical backfill of features for model training.

    python scripts/backfill_historical_data.py \
        --city Karachi --start-date 2024-01-01 --end-date 2026-01-01 \
        [--batch-days 30] [--dry-run] [--sample] [--no-resume] [--max-retries 3]

Properties:
  * Batched by ``--batch-days`` (provider rate-limit friendly).
  * Resumable — completed batches are recorded and skipped on re-run.
  * Idempotent / de-duplicated — rows upserted on (city_id, timestamp).
  * Failed intervals recorded and retried (up to ``--max-retries``).
  * Writes a backfill summary + a missing-data report.
  * ``--dry-run`` fetches/validates but does not write.
  * ``--sample`` uses the deterministic synthetic generator when live history is
    unavailable (clearly tagged; never presented as real observations).
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta

from _common import bootstrap, logger, parse_date, resolve_location
from pearls_aqi.api_clients.sample_provider import (
    SampleAirQualityProvider,
    SampleWeatherProvider,
)
from pearls_aqi.data.validation import missing_data_report
from pearls_aqi.pipelines import FeaturePipeline
from pearls_aqi.storage import get_feature_store
from pearls_aqi.storage.base import FEATURES_GROUP
from pearls_aqi.utils.io import read_json, write_json
from pearls_aqi.utils.timeutils import now_utc


def _batches(start: datetime, end: datetime, batch_days: int) -> list[tuple[datetime, datetime]]:
    out, cursor = [], start
    while cursor < end:
        b_end = min(cursor + timedelta(days=batch_days), end)
        out.append((cursor, b_end))
        cursor = b_end
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill historical features.")
    parser.add_argument("--city", default=None)
    parser.add_argument("--start-date", type=parse_date, required=True)
    parser.add_argument("--end-date", type=parse_date, required=True)
    parser.add_argument("--batch-days", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample", action="store_true", help="Use synthetic sample data.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore prior progress.")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    config = bootstrap()
    location = resolve_location(config, args.city)
    batch_days = args.batch_days or config.providers.batch_days
    if args.end_date <= args.start_date:
        logger.error("--end-date must be after --start-date")
        return 2

    if args.sample:
        pipeline = FeaturePipeline(
            config, SampleAirQualityProvider(), SampleWeatherProvider(), get_feature_store(config)
        )
        logger.info("Backfill running in SAMPLE mode (synthetic data).")
    else:
        pipeline = FeaturePipeline(config)

    state_path = f"artifacts/backfill/{location.city_id}/state.json"
    completed: set[str] = set()
    if not args.no_resume:
        try:
            completed = set(read_json(state_path).get("completed", []))
            if completed:
                logger.info("Resuming: %d batches already completed.", len(completed))
        except Exception:
            completed = set()

    batches = _batches(args.start_date, args.end_date, batch_days)
    total_fetched = total_written = 0
    failed: list[dict] = []
    per_batch: list[dict] = []

    for i, (b_start, b_end) in enumerate(batches, 1):
        key = f"{b_start.date()}_{b_end.date()}"
        if key in completed and not args.no_resume:
            logger.info("[%d/%d] skip %s (done)", i, len(batches), key)
            continue
        ok = False
        for attempt in range(1, args.max_retries + 1):
            try:
                summary = pipeline.run(location, b_start, b_end, write=not args.dry_run)
                total_fetched += summary.fetched
                total_written += summary.written
                per_batch.append({"batch": key, **summary.as_dict()})
                completed.add(key)
                ok = True
                logger.info(
                    "[%d/%d] %s fetched=%d written=%d",
                    i,
                    len(batches),
                    key,
                    summary.fetched,
                    summary.written,
                )
                break
            except Exception as exc:
                logger.warning(
                    "[%d/%d] %s attempt %d failed: %s", i, len(batches), key, attempt, exc
                )
                time.sleep(min(5, attempt))
        if not ok:
            failed.append({"batch": key, "start": b_start.isoformat(), "end": b_end.isoformat()})
        if not args.dry_run:
            write_json({"completed": sorted(completed), "failed": failed}, state_path)

    # Missing-data report over the stored range.
    missing = []
    if not args.dry_run:
        stored = get_feature_store(config).read_features(
            FEATURES_GROUP, start=args.start_date, end=args.end_date, city_id=location.city_id
        )
        if stored is not None and not stored.empty:
            missing = missing_data_report(stored).head(20).to_dict(orient="records")

    stamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    summary_doc = {
        "city": location.city,
        "start": args.start_date.isoformat(),
        "end": args.end_date.isoformat(),
        "batch_days": batch_days,
        "dry_run": args.dry_run,
        "sample": args.sample,
        "batches_total": len(batches),
        "batches_completed": len(completed),
        "batches_failed": failed,
        "total_fetched": total_fetched,
        "total_written": total_written,
        "missing_data_top20": missing,
        "generated_at": now_utc().isoformat(),
    }
    write_json(summary_doc, f"artifacts/backfill/{location.city_id}/summary_{stamp}.json")
    write_json(summary_doc, f"artifacts/backfill/{location.city_id}/summary_latest.json")

    print(
        f"Backfill {'(dry-run) ' if args.dry_run else ''}for {location.city}: "
        f"{len(completed)}/{len(batches)} batches, written={total_written}, "
        f"failed={len(failed)}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
