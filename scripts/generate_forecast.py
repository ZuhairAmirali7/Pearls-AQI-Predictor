#!/usr/bin/env python
"""Generate a 72-hour AQI forecast from the approved model.

    python scripts/generate_forecast.py --city Karachi
    python scripts/generate_forecast.py --city Karachi --output forecast.json

Also evaluates hazard alerts on the forecast and prints them.
"""

from __future__ import annotations

import argparse
import json

from _common import bootstrap, logger, resolve_location
from pearls_aqi.alerts import generate_alerts
from pearls_aqi.exceptions import PearlsError
from pearls_aqi.forecasting import ForecastService
from pearls_aqi.utils.io import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a 72h forecast.")
    parser.add_argument("--city", default=None)
    parser.add_argument("--model-version", type=int, default=None)
    parser.add_argument("--output", default=None, help="Write forecast JSON to this path.")
    parser.add_argument("--no-weather", action="store_true", help="Skip fetching future weather.")
    args = parser.parse_args()

    config = bootstrap()
    location = resolve_location(config, args.city)
    service = ForecastService(config)

    try:
        result = service.generate(
            city=location.city,
            model_version=args.model_version,
            fetch_weather=not args.no_weather,
        )
    except PearlsError as exc:
        logger.error("Forecast failed: %s", exc)
        return 1

    payload = result.as_dict()
    alerts = generate_alerts(result.forecast, config.alerts, city=location.city)
    payload["alerts"] = [a.as_dict() for a in alerts]

    if args.output:
        write_json(payload, args.output)
        print(f"Wrote forecast to {args.output}")
    else:
        print(json.dumps(payload, indent=2))

    peak = result.forecast["predicted_aqi"].max()
    logger.info(
        "Forecast for %s: %d hours, peak AQI %.0f, %d alert(s).",
        location.city,
        len(result.forecast),
        peak,
        len(alerts),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
