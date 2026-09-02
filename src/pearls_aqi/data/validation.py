"""Data validation for raw and processed observation frames.

Produces a structured :class:`ValidationReport` (never raises for non-critical
issues) so pipelines can *fail clearly on critical errors* while *warning and
continuing* on recoverable ones. Critical failures: missing required columns,
duplicate primary keys, non-chronological order, empty frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pearls_aqi.data.schemas import (
    PHYSICAL_BOUNDS,
    PRIMARY_KEY,
)
from pearls_aqi.exceptions import ValidationError


@dataclass
class ValidationIssue:
    level: str  # "error" | "warning"
    check: str
    message: str
    count: int = 0


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    rows: int = 0

    def add(self, level: str, check: str, message: str, count: int = 0) -> None:
        self.issues.append(ValidationIssue(level, check, message, count))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if not self.ok:
            details = "; ".join(f"[{i.check}] {i.message}" for i in self.errors)
            raise ValidationError(f"Validation failed ({len(self.errors)} error(s)): {details}")

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "ok": self.ok,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "issues": [
                {"level": i.level, "check": i.check, "message": i.message, "count": i.count}
                for i in self.issues
            ],
        }


def validate_observations(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
    expect_hourly: bool = True,
    allow_empty: bool = False,
) -> ValidationReport:
    """Validate a merged observation frame; return a report."""
    report = ValidationReport(rows=len(df))

    if df.empty:
        level = "warning" if allow_empty else "error"
        report.add(level, "non_empty", "Observation frame is empty.")
        return report

    # Required columns.
    required = required_columns or [*PRIMARY_KEY]
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.add("error", "required_columns", f"Missing required columns: {missing}")
        return report  # further checks assume these exist

    # Timestamp dtype + timezone.
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        report.add("error", "timestamp_dtype", "`timestamp` must be datetime.")
        return report

    # Duplicate primary keys.
    dup = df.duplicated(subset=PRIMARY_KEY).sum()
    if dup:
        report.add(
            "error", "unique_primary_key", f"{dup} duplicate (city_id, timestamp) rows.", int(dup)
        )

    # Chronological order (per city).
    for _, g in df.groupby("city_id"):
        ts = g["timestamp"]
        if not ts.is_monotonic_increasing:
            report.add("error", "chronological", "Rows are not sorted by timestamp within a city.")
            break

    # Physical bounds.
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        mask = pd.Series(False, index=series.index)
        if lo is not None:
            mask |= series < lo
        if hi is not None:
            mask |= series > hi
        n_bad = int(mask.sum())
        if n_bad:
            report.add(
                "warning",
                "physical_bounds",
                f"{n_bad} `{col}` values outside [{lo}, {hi}].",
                n_bad,
            )

    # Missing-value rate per numeric column.
    for col in df.columns:
        if col in PRIMARY_KEY:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            frac = float(df[col].isna().mean())
            if frac > 0.5:
                report.add(
                    "warning",
                    "missing_rate",
                    f"`{col}` is {frac:.0%} missing.",
                    int(df[col].isna().sum()),
                )

    # Timestamp continuity (hourly gaps).
    if expect_hourly:
        for city, g in df.groupby("city_id"):
            ts = g["timestamp"].sort_values()
            if len(ts) < 2:
                continue
            full = pd.date_range(ts.iloc[0], ts.iloc[-1], freq="h", tz="UTC")
            missing_hours = len(full) - ts.nunique()
            if missing_hours > 0:
                report.add(
                    "warning",
                    "timestamp_continuity",
                    f"{missing_hours} missing hourly timestamps for city_id={city}.",
                    int(missing_hours),
                )

    return report


def missing_data_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts/fractions — used by the backfill summary."""
    if df.empty:
        return pd.DataFrame(columns=["column", "missing", "fraction"])
    rows = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        rows.append({"column": col, "missing": n_missing, "fraction": n_missing / len(df)})
    return pd.DataFrame(rows).sort_values("fraction", ascending=False, ignore_index=True)


def find_outliers_iqr(series: pd.Series, k: float = 3.0) -> pd.Series:
    """Boolean mask of IQR-based outliers (k*IQR beyond Q1/Q3)."""
    s = pd.to_numeric(series, errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        return pd.Series(False, index=series.index)
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)
