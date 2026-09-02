"""Provider interfaces and a resilient HTTP base client.

All providers return **UTC-timestamped** pandas DataFrames with a ``timestamp``
column plus a ``data_source`` column; other columns follow the canonical schema
(:mod:`pearls_aqi.data.schemas`). Missing values stay ``NaN`` — never zero-filled.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import pandas as pd
import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pearls_aqi.data.domain import Location
from pearls_aqi.exceptions import (
    ProviderRequestError,
    ProviderResponseError,
    RateLimitError,
)
from pearls_aqi.utils.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class AirQualityProvider(Protocol):
    """Fetches pollutant concentrations for a location."""

    name: str

    def fetch_current(self, location: Location) -> pd.DataFrame: ...

    def fetch_historical(
        self, location: Location, start: datetime, end: datetime
    ) -> pd.DataFrame: ...


@runtime_checkable
class WeatherProvider(Protocol):
    """Fetches weather observations and forecasts for a location."""

    name: str

    def fetch_current(self, location: Location) -> pd.DataFrame: ...

    def fetch_historical(
        self, location: Location, start: datetime, end: datetime
    ) -> pd.DataFrame: ...

    def fetch_forecast(self, location: Location, horizon_hours: int) -> pd.DataFrame: ...


class BaseHTTPClient:
    """Shared HTTP client with timeout, retry/backoff, and rate-limit handling."""

    def __init__(
        self,
        timeout: float = 20.0,
        max_retries: int = 4,
        backoff_seconds: float = 1.5,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._session = session or requests.Session()

    def get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET ``url`` with retries; return parsed JSON or raise a ProviderError."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=self.backoff_seconds, min=self.backoff_seconds, max=30
            ),
            retry=retry_if_exception_type((requests.exceptions.RequestException, RateLimitError)),
            before_sleep=before_sleep_log(logger, 30),  # logging.WARNING == 30
        )
        def _do_request() -> dict[str, Any]:
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
            except requests.exceptions.RequestException as exc:
                logger.warning("Request error to %s: %s", url, exc)
                raise
            if resp.status_code == 429:
                raise RateLimitError(f"Rate limited by {url} (HTTP 429).")
            if resp.status_code >= 500:
                raise ProviderRequestError(f"Server error {resp.status_code} from {url}.")
            if resp.status_code >= 400:
                raise ProviderResponseError(
                    f"Client error {resp.status_code} from {url}: {resp.text[:200]}"
                )
            try:
                return resp.json()
            except ValueError as exc:
                raise ProviderResponseError(f"Non-JSON response from {url}: {exc}") from exc

        try:
            return _do_request()
        except RateLimitError:
            raise
        except requests.exceptions.RequestException as exc:
            raise ProviderRequestError(f"Failed to reach {url} after retries: {exc}") from exc


def hourly_frame_from_arrays(
    times: list[str],
    variables: dict[str, list[Any]],
    source: str,
) -> pd.DataFrame:
    """Build a UTC-timestamped frame from Open-Meteo-style parallel arrays."""
    if not times:
        raise ProviderResponseError(f"Empty 'time' array from provider {source}.")
    df = pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True)})
    for col, values in variables.items():
        df[col] = values
    df["data_source"] = source
    return df
