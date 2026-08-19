"""
Validation
==========

Completeness / consistency / reasonableness checks between cleaned data and
the feature layer. Ported from ``f1pipeline/validation.py`` — logic and
severities unchanged; see ``MIGRATION_MAP.md`` for why this lives in
``data/`` even though the target architecture sketch didn't list it
separately (it's the natural home: a validation check is "is this cleaned
batch trustworthy," not a modeling concern).

Three categories, in order of cost:

1. **Completeness** — did we get all the sensors, for all the laps, for
   every driver?
2. **Consistency** — do values that must relate to each other actually do
   so (a cumulative gap that only grows, a stint's tyre age that resets on
   the right lap)?
3. **Reasonableness** — are values within physically plausible bounds?

A failing check does not raise; it's recorded as a :class:`CheckResult` with
a severity, and :attr:`ValidationReport.passed` rolls them up — any
``ERROR`` fails the batch, a ``WARNING`` does not.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


class Status(str, Enum):
    """Severity of a single check. An ``ERROR`` fails the batch; a ``WARNING`` is recorded but does not."""

    PASS = "PASS"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class CheckResult:
    """One check's outcome: its name, category, severity, message, and how many rows it affected."""

    name: str
    category: str
    status: Status
    message: str
    n_affected: int = 0
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "category": self.category, "status": self.status.value,
            "message": self.message, "n_affected": self.n_affected, "details": self.details,
        }


@dataclass
class ValidationReport:
    """The rolled-up result of a validation run; :attr:`passed` is True unless any check is an ERROR."""

    generated_at: datetime
    n_rows: int
    results: list[CheckResult] = field(default_factory=list)

    @property
    def errors(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.ERROR]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.WARNING]

    @property
    def passed(self) -> bool:
        """Whether the batch clears the publication gate — no ERROR-level failures."""
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at.isoformat(), "n_rows": self.n_rows,
            "passed": self.passed, "n_errors": len(self.errors), "n_warnings": len(self.warnings),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        header = "PASS" if self.passed else "FAIL"
        lines = [
            f"[{header}] validated {self.n_rows:,} rows, "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        ]
        for result in self.results:
            if result.status is not Status.PASS:
                lines.append(f"  [{result.status.value:7s}] {result.category}/{result.name}: {result.message}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ValidationConfig:
    """Thresholds for :func:`validate_laps`. Override per environment, not per call site."""

    required_columns: tuple[str, ...] = (
        "session_key", "driver_number", "lap_number", "lap_time_seconds", "tyre_compound", "tyre_life",
    )
    max_null_rate: dict[str, float] = field(default_factory=lambda: {
        "lap_time_seconds": 0.05, "tyre_compound": 0.05, "tyre_life": 0.05, "position": 0.05,
    })
    default_max_null_rate: float = 0.10
    expected_drivers: int | None = None
    min_drivers: int = 2
    lap_time_bounds: tuple[float, float] = (55.0, 150.0)
    degradation_rate_bounds: tuple[float, float] = (-0.5, 0.5)
    pit_delta_bounds: tuple[float, float] = (1.5, 90.0)
    reasonableness_n_sigma: float = 2.0
    reasonableness_min_group_size: int = 5


def _add(
    results: list[CheckResult], name: str, category: str, status: Status, message: str,
    n_affected: int = 0, **details,
) -> None:
    results.append(CheckResult(
        name=name, category=category, status=status, message=message,
        n_affected=n_affected, details=details,
    ))


def check_completeness(frame: pd.DataFrame, config: ValidationConfig) -> list[CheckResult]:
    """Check required columns, per-column null rates, per-session driver coverage, and lap-sequence gaps."""
    results: list[CheckResult] = []

    missing_cols = [c for c in config.required_columns if c not in frame.columns]
    if missing_cols:
        _add(results, "required_columns", "completeness", Status.ERROR,
             f"Missing required columns: {missing_cols}", n_affected=len(missing_cols), columns=missing_cols)
        return results
    _add(results, "required_columns", "completeness", Status.PASS, "All required columns present")

    for column, threshold in config.max_null_rate.items():
        if column not in frame.columns:
            continue
        null_rate = float(frame[column].isna().mean()) if len(frame) else 0.0
        status = Status.PASS if null_rate <= threshold else Status.ERROR
        _add(results, f"null_rate:{column}", "completeness", status,
             f"{column}: {null_rate * 100:.2f}% null (threshold {threshold * 100:.1f}%)",
             n_affected=int(frame[column].isna().sum()), null_rate=null_rate, threshold=threshold)

    other_cols = [
        c for c in frame.columns
        if c not in config.max_null_rate and not pd.api.types.is_object_dtype(frame[c])
    ]
    for column in other_cols:
        if column in config.required_columns:
            continue
        null_rate = float(frame[column].isna().mean()) if len(frame) else 0.0
        if null_rate > config.default_max_null_rate:
            _add(results, f"null_rate:{column}", "completeness", Status.WARNING,
                 f"{column}: {null_rate * 100:.2f}% null (default threshold {config.default_max_null_rate * 100:.1f}%)",
                 n_affected=int(frame[column].isna().sum()), null_rate=null_rate)

    if {"session_key", "driver_number", "lap_number"}.issubset(frame.columns):
        for session_key, session in frame.groupby("session_key", dropna=False):
            n_drivers = session["driver_number"].nunique(dropna=True)
            expected = config.expected_drivers
            if expected is not None and n_drivers < expected:
                _add(results, f"driver_coverage:{session_key}", "completeness", Status.WARNING,
                     f"session {session_key}: {n_drivers} drivers present, expected {expected}",
                     n_affected=expected - n_drivers, session_key=str(session_key))
            elif n_drivers < config.min_drivers:
                _add(results, f"driver_coverage:{session_key}", "completeness", Status.ERROR,
                     f"session {session_key}: only {n_drivers} driver(s) present (minimum {config.min_drivers})",
                     n_affected=n_drivers, session_key=str(session_key))

            gaps = _find_lap_number_gaps(session)
            if gaps:
                described = "; ".join(f"driver {drv}: laps {lo}-{hi} missing" for drv, lo, hi in gaps[:5])
                _add(results, f"lap_sequence_gaps:{session_key}", "completeness", Status.WARNING,
                     f"session {session_key}: {len(gaps)} lap-sequence gap(s) — {described}",
                     n_affected=len(gaps), session_key=str(session_key), gaps=gaps[:20])

    return results


def _find_lap_number_gaps(session: pd.DataFrame) -> list[tuple[int, int, int]]:
    gaps: list[tuple[int, int, int]] = []
    for driver, driver_laps in session.groupby("driver_number", dropna=False):
        laps = sorted(int(n) for n in driver_laps["lap_number"].dropna().unique())
        if len(laps) < 2:
            continue
        for previous, current in zip(laps, laps[1:]):
            if current - previous > 1:
                gaps.append((int(driver), previous + 1, current - 1))
    return gaps


def check_consistency(frame: pd.DataFrame, config: ValidationConfig) -> list[CheckResult]:
    """Check cross-field invariants: monotonic gap-to-leader, tyre-age reset per stint, strictly increasing
    lap numbers, and unique position per lap."""
    results: list[CheckResult] = []
    group_cols = [c for c in ("session_key", "driver_number") if c in frame.columns]

    if "gap_to_leader" in frame.columns and group_cols and "lap_number" in frame.columns:
        ordered = frame.sort_values(group_cols + ["lap_number"])
        decreasing = ordered.groupby(group_cols, sort=False)["gap_to_leader"].diff() < -1e-6
        n_bad = int(decreasing.fillna(False).sum())
        status = Status.PASS if n_bad == 0 else Status.ERROR
        _add(results, "monotonic_cumulative_gap", "consistency", status,
             "gap_to_leader is non-decreasing within each driver's session" if n_bad == 0
             else f"gap_to_leader decreased on {n_bad} lap(s) — cumulative gap must never shrink",
             n_affected=n_bad)

    if "tyre_age" in frame.columns and "stint_number" in frame.columns and group_cols:
        # Keyed off a stint_number change, not is_pit_lap: FastF1 marks *both*
        # the in-lap and the out-lap as is_pit_lap, so using it directly would
        # also demand tyre_age == 0 on the lap after the out-lap — a false
        # positive on every single pit stop. See docs/decisions/.
        ordered = frame.sort_values(group_cols + ["lap_number"])
        stint_changed = ordered.groupby(group_cols, sort=False)["stint_number"].diff().fillna(0) != 0
        age = ordered["tyre_age"]
        bad = stint_changed & (age != 0)
        n_bad = int(bad.sum())
        status = Status.PASS if n_bad == 0 else Status.ERROR
        _add(results, "tyre_age_resets_after_pit", "consistency", status,
             "tyre_age resets to 0 on the first lap of every stint" if n_bad == 0
             else f"tyre_age failed to reset to 0 on {n_bad} stint-change lap(s)",
             n_affected=n_bad)

    if group_cols and "lap_number" in frame.columns:
        ordered = frame.sort_values(group_cols + ["lap_number"])
        deltas = ordered.groupby(group_cols, sort=False)["lap_number"].diff()
        bad = deltas.notna() & (deltas <= 0)
        n_bad = int(bad.sum())
        status = Status.PASS if n_bad == 0 else Status.ERROR
        _add(results, "lap_number_strictly_increasing", "consistency", status,
             "lap_number strictly increases within each driver's session" if n_bad == 0
             else f"lap_number failed to strictly increase on {n_bad} row(s) (duplicate or out-of-order lap)",
             n_affected=n_bad)

    if "position" in frame.columns and "session_key" in frame.columns and "lap_number" in frame.columns:
        counts = frame.dropna(subset=["position"]).groupby(["session_key", "lap_number"])["position"]
        duplicated = counts.apply(lambda s: s.duplicated().sum())
        n_bad = int(duplicated.sum())
        status = Status.PASS if n_bad == 0 else Status.WARNING
        _add(results, "unique_position_per_lap", "consistency", status,
             "position is unique per lap within each session" if n_bad == 0
             else f"{n_bad} duplicate position value(s) on a shared lap — check safety-car/pit-lane laps",
             n_affected=n_bad)

    return results


def _bounds_check(frame: pd.DataFrame, column: str, bounds: tuple[float, float]) -> tuple[int, int, int]:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    low, high = bounds
    return int((values < low).sum()), int((values > high).sum()), len(values)


def check_reasonableness(frame: pd.DataFrame, config: ValidationConfig) -> list[CheckResult]:
    """Check values against physically plausible bounds and flag lap times beyond nσ of each driver's median."""
    results: list[CheckResult] = []

    bounded = {
        "lap_time_seconds": config.lap_time_bounds,
        "degradation_rate": config.degradation_rate_bounds,
        "pit_delta": config.pit_delta_bounds,
    }
    for column, bounds in bounded.items():
        if column not in frame.columns:
            continue
        n_low, n_high, n_total = _bounds_check(frame, column, bounds)
        n_bad = n_low + n_high
        status = Status.PASS if n_bad == 0 else Status.ERROR
        low, high = bounds
        _add(results, f"bounds:{column}", "reasonableness", status,
             f"{column}: all {n_total} value(s) within [{low}, {high}]" if n_bad == 0
             else f"{column}: {n_bad}/{n_total} value(s) outside [{low}, {high}] ({n_low} below, {n_high} above)",
             n_affected=n_bad, bounds=list(bounds))

    if "lap_time_seconds" in frame.columns:
        group_cols = [c for c in ("session_key", "driver_number") if c in frame.columns]
        n_sigma = config.reasonableness_n_sigma
        n_bad = 0
        n_checked = 0
        for _, group in (frame.groupby(group_cols, dropna=False) if group_cols else [((), frame)]):
            values = pd.to_numeric(group["lap_time_seconds"], errors="coerce").dropna()
            if len(values) < config.reasonableness_min_group_size:
                continue
            median = values.median()
            mad = (values - median).abs().median()
            if mad <= 0:
                continue
            sigma = mad * 1.4826
            n_bad += int(((values - median).abs() > n_sigma * sigma).sum())
            n_checked += len(values)
        status = Status.PASS if n_bad == 0 else Status.WARNING
        message = (
            f"lap_time_seconds: all {n_checked} checked value(s) within {n_sigma}σ of their driver's median"
            if n_bad == 0 else
            f"lap_time_seconds: {n_bad}/{n_checked} value(s) beyond {n_sigma}σ of their driver's "
            "median — review before use"
        )
        _add(results, "lap_time_within_driver_sigma", "reasonableness", status, message,
             n_affected=n_bad, n_sigma=n_sigma)

    return results


_CHECK_STAGES: tuple[tuple[str, Callable[[pd.DataFrame, ValidationConfig], list[CheckResult]]], ...] = (
    ("completeness", check_completeness),
    ("consistency", check_consistency),
    ("reasonableness", check_reasonableness),
)


def validate_laps(frame: pd.DataFrame, config: ValidationConfig | None = None) -> ValidationReport:
    """
    Run completeness, consistency and reasonableness checks in order.

    Checks whose columns aren't present are skipped, not failed — this
    validates what was asked of it, not which metrics were computed.
    """
    config = config or ValidationConfig()
    report = ValidationReport(generated_at=datetime.now(timezone.utc), n_rows=len(frame))
    for _, check_fn in _CHECK_STAGES:
        report.results.extend(check_fn(frame, config))
    logger.info(report.summary())
    return report
