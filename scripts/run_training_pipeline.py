#!/usr/bin/env python
"""Train, compare, select, register, and conditionally promote a model.

    python scripts/run_training_pipeline.py --city Karachi
    python scripts/run_training_pipeline.py --city Karachi --no-tune \
        --models ridge random_forest --horizon 72

Uses chronological validation (never shuffled). Prints the model-comparison
table and the promotion decision; the full report is written under
artifacts/reports/<city_id>/.
"""

from __future__ import annotations

import argparse

from _common import bootstrap, logger, resolve_location
from pearls_aqi.exceptions import PearlsError
from pearls_aqi.pipelines import TrainingPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the training pipeline.")
    parser.add_argument("--city", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model names.")
    parser.add_argument("--no-tune", action="store_true", help="Disable hyperparameter search.")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap training rows (debug/CI).")
    args = parser.parse_args()

    config = bootstrap()
    location = resolve_location(config, args.city)
    pipeline = TrainingPipeline(config)

    try:
        report = pipeline.run(
            location,
            horizon=args.horizon,
            models=args.models,
            tune=(False if args.no_tune else None),
            max_rows=args.max_rows,
        )
    except PearlsError as exc:
        logger.error("Training failed: %s", exc)
        return 1

    print(report.to_markdown())
    print(
        f"\nRegistered {report.model_name} v{report.registered_version} "
        f"as {report.registered_status}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
