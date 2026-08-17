"""
Cleaning: normalize raw laps to the contract, then repair bounded sensor gaps
===============================================================================

Two responsibilities, kept in one module because they run back-to-back on
every batch and share the same "never fabricate a value" discipline:

1. **Normalize** — units to seconds, tyre compound to
   :class:`f1_pit_window.data.contracts.TyreCompound`, driver identity fixed,
   implausible lap times flagged (never dropped or zero-filled).
2. **Impute** — bounded sensor-gap repair. A gap under ``max_gap_seconds`` is
   forward-filled and marked; a longer gap is left ``NaN`` and marked as a
   sensor failure. Never guessed at, never silently zero-filled.

Ported from ``f1pipeline/normalize.py`` and ``f1pipeline/imputation.py`` —
merged into one module because the target architecture (see
``MIGRATION_MAP.md``) treats "clean the data" as one pipeline stage, not two.
Behavior is unchanged from the source modules; see ``docs/decisions/`` for the
bugs that were caught and fixed while the original modules were built.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from f1_pit_window.data.contracts import (
    COMPOUND_ALIASES,
    DURATION_COLUMNS,
    FASTF1_COLUMN_MAP,
    TyreCompound,
)

logger = logging.getLogger(__name__)

# Robust-sigma scaling: for normally distributed data, sigma ≈ 1.4826 × MAD.
_MAD_TO_SIGMA = 1.4826

# "1:32.456" / "01:32.456" / "32.456"
_CLOCK_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2})(?:\.(\d{1,6}))?\s*$")


# ─────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────


def normalize_compound(value: object) -> TyreCompound:
    """
    Map any raw compound spelling onto the canonical vocabulary.

    Unrecognised values normalize to :attr:`TyreCompound.UNKNOWN` and are
    logged, rather than raising — one novel spelling should not abort a
    season's ingest, but it must not pass silently either.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return TyreCompound.UNKNOWN
    if isinstance(value, TyreCompound):
        return value

    key = str(value).strip().upper()
    if not key:
        return TyreCompound.UNKNOWN

    compound = COMPOUND_ALIASES.get(key)
    if compound is None:
        logger.warning("Unrecognised tyre compound %r → UNKNOWN", value)
        return TyreCompound.UNKNOWN
    return compound


def to_seconds(value: object, unit: str = "auto") -> float:
    """
    Convert a duration of any incoming shape to a float number of seconds.

    Handled inputs: ``timedelta``/``Timedelta``/``timedelta64``, clock strings
    such as ``"1:32.456"``, and plain numbers. ``unit`` controls how *numeric*
    input is read (seconds by default) — guessing units from magnitude is
    exactly the ad-hoc behavior this layer exists to eliminate.

    Raises:
        ValueError: If ``unit`` is not one of ``auto``/``s``/``ms``/``us``/``ns``.
    """
    scale = {"auto": 1.0, "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}.get(unit)
    if scale is None:
        raise ValueError(f"Unsupported unit {unit!r}; expected one of auto, s, ms, us, ns")

    if value is None or value is pd.NaT:
        return float("nan")
    if isinstance(value, (pd.Timedelta, _dt.timedelta, np.timedelta64)):
        return float(pd.Timedelta(value).total_seconds())
    if isinstance(value, str):
        match = _CLOCK_RE.match(value)
        if match:
            minutes, seconds, frac = match.groups()
            total = int(minutes or 0) * 60 + int(seconds)
            if frac:
                total += int(frac) / (10 ** len(frac))
            return float(total)
        try:
            return float(value) * scale
        except ValueError:
            logger.warning("Unparseable duration %r → nan", value)
            return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return float("nan") if np.isnan(numeric) else numeric * scale

    logger.warning("Unparseable duration of type %s → nan", type(value).__name__)
    return float("nan")


def _normalize_duration_column(series: pd.Series, unit: str = "auto") -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds().astype(float)
    if pd.api.types.is_numeric_dtype(series) and unit in ("auto", "s"):
        return series.astype(float)
    return series.map(lambda value: to_seconds(value, unit=unit)).astype(float)


def flag_outliers(
    frame: pd.DataFrame,
    column: str,
    group_cols: Sequence[str] = ("session_key", "driver_number"),
    n_sigma: float = 3.0,
    min_group_size: int = 5,
) -> pd.Series:
    """
    Flag values implausible relative to their own group (median ± n_sigma
    robust-sigma via MAD × 1.4826), never against a global distribution — a
    driver who is genuinely slower all race must not be flagged for it.
    """
    flags = pd.Series(False, index=frame.index, name=f"{column}_outlier")
    if column not in frame.columns:
        return flags

    present = [col for col in group_cols if col in frame.columns]
    groups: Iterable[tuple] = frame.groupby(list(present), dropna=False) if present else [((), frame)]

    for _, group in groups:
        values = pd.to_numeric(group[column], errors="coerce")
        valid = values.dropna()
        if len(valid) < min_group_size:
            continue
        median = valid.median()
        mad = (valid - median).abs().median()
        if mad <= 0:
            continue
        deviation = (values - median).abs() / (mad * _MAD_TO_SIGMA)
        flags.loc[group.index] = (deviation > n_sigma).fillna(False)

    return flags


def normalize_laps(
    frame: pd.DataFrame,
    session_key: str | None = None,
    flag_lap_time_outliers: bool = True,
    n_sigma: float = 3.0,
) -> pd.DataFrame:
    """
    Bring a raw FastF1 lap frame into the canonical contract.

    Missing values are left as ``NaN`` — filling them is :func:`impute_gaps`'s
    job, which records *what* it filled. This function never fabricates data.

    Raises:
        ValueError: If ``session_key`` is required (absent from ``frame``)
            but not supplied.
    """
    df = frame.copy()
    df = df.rename(columns={k: v for k, v in FASTF1_COLUMN_MAP.items() if k in df.columns})

    if "session_key" not in df.columns:
        if session_key is None:
            raise ValueError("session_key must be provided when the frame has no session_key column")
        df["session_key"] = session_key

    for column in DURATION_COLUMNS:
        if column in df.columns:
            df[column] = _normalize_duration_column(df[column])

    if "tyre_compound" in df.columns:
        df["tyre_compound_raw"] = df["tyre_compound"]
        df["tyre_compound"] = df["tyre_compound"].map(normalize_compound).map(lambda c: c.value)

    if "driver_number" in df.columns:
        df["driver_number"] = pd.to_numeric(df["driver_number"], errors="coerce").astype("Int64")
    if "driver_code" in df.columns:
        df["driver_code"] = df["driver_code"].astype("string").str.strip().str.upper().str[:3]
    for column in ("lap_number", "tyre_life", "stint_number", "position"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    if "team_name" in df.columns:
        df["team_name"] = df["team_name"].astype("string").str.strip()

    pit_columns = [col for col in ("pit_in_time", "pit_out_time") if col in df.columns]
    if pit_columns:
        df["is_pit_lap"] = df[pit_columns].notna().any(axis=1)
    elif "is_pit_lap" in df.columns:
        df["is_pit_lap"] = df["is_pit_lap"].fillna(False).astype(bool)
    else:
        df["is_pit_lap"] = False

    if "rainfall" in df.columns:
        df["rainfall"] = df["rainfall"].fillna(False).astype(bool)

    if flag_lap_time_outliers and "lap_time_seconds" in df.columns:
        df["lap_time_outlier"] = flag_outliers(df, "lap_time_seconds", n_sigma=n_sigma)
        flagged = int(df["lap_time_outlier"].sum())
        if flagged:
            logger.info("Flagged %d/%d laps as lap-time outliers for manual review", flagged, len(df))

    sort_cols = [c for c in ("session_key", "driver_number", "lap_number") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────
# Imputation
# ─────────────────────────────────────────────────────────────

FORWARD_FILLED = "forward_filled"
SENSOR_FAILURE = "sensor_failure"


@dataclass(frozen=True)
class GapEvent:
    """One contiguous run of missing samples, and what was done about it."""

    column: str
    group: tuple
    n_samples: int
    duration_seconds: float
    action: str
    reason: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["group"] = [str(part) for part in self.group]
        return payload


@dataclass
class ImputationReport:
    """Audit trail for an :func:`impute_gaps` run."""

    max_gap_seconds: float
    events: list[GapEvent] = field(default_factory=list)
    completeness_before: dict[str, float] = field(default_factory=dict)
    completeness_after: dict[str, float] = field(default_factory=dict)

    @property
    def filled_samples(self) -> int:
        return sum(e.n_samples for e in self.events if e.action == FORWARD_FILLED)

    @property
    def failed_samples(self) -> int:
        return sum(e.n_samples for e in self.events if e.action == SENSOR_FAILURE)

    def to_dict(self) -> dict:
        return {
            "max_gap_seconds": self.max_gap_seconds,
            "filled_samples": self.filled_samples,
            "failed_samples": self.failed_samples,
            "completeness_before": self.completeness_before,
            "completeness_after": self.completeness_after,
            "events": [e.to_dict() for e in self.events],
        }


def _completeness(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, float]:
    if len(frame) == 0:
        return {column: 100.0 for column in columns}
    return {column: float(frame[column].notna().mean() * 100.0) for column in columns}


def _missing_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for position, missing in enumerate(mask):
        if missing and start is None:
            start = position
        elif not missing and start is not None:
            runs.append((start, position))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _median_interval(times: np.ndarray) -> float:
    if len(times) < 2:
        return float("nan")
    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if len(diffs) else float("nan")


def impute_gaps(
    frame: pd.DataFrame,
    columns: Sequence[str],
    time_col: str | None = None,
    sample_interval_seconds: float | None = None,
    max_gap_seconds: float = 1.0,
    group_cols: Sequence[str] = ("session_key", "driver_number"),
) -> tuple[pd.DataFrame, ImputationReport]:
    """
    Forward-fill short sensor gaps; mark long ones as sensor failures.

    A gap at the *start* of a group is always a sensor failure regardless of
    length — there is no prior value to carry forward, and back-filling would
    leak future information backwards in time.

    Raises:
        ValueError: If ``max_gap_seconds`` is negative.
    """
    if max_gap_seconds < 0:
        raise ValueError(f"max_gap_seconds must be >= 0, got {max_gap_seconds}")

    df = frame.copy()
    present = [column for column in columns if column in df.columns]
    for missing in set(columns) - set(present):
        logger.warning("Column %r not present; skipping imputation for it", missing)

    report = ImputationReport(max_gap_seconds=max_gap_seconds)
    if not present:
        return df, report

    report.completeness_before = _completeness(df, present)

    for column in present:
        df[f"{column}_imputed"] = False
        df[f"{column}_sensor_failure"] = False

    group_keys = [column for column in group_cols if column in df.columns]
    groups = list(df.groupby(group_keys, dropna=False, sort=False)) if group_keys else [((), df)]

    for key, group in groups:
        group_id = key if isinstance(key, tuple) else (key,)
        index = group.index

        if sample_interval_seconds is not None:
            interval = float(sample_interval_seconds)
        elif time_col is not None and time_col in group.columns:
            interval = _median_interval(pd.to_numeric(group[time_col], errors="coerce").to_numpy(dtype=float))
        else:
            interval = float("nan")

        for column in present:
            values = group[column]
            mask = values.isna().to_numpy()
            if not mask.any():
                continue

            for start, end in _missing_runs(mask):
                n_samples = end - start
                duration = n_samples * interval if np.isfinite(interval) else float("nan")
                run_index = index[start:end]

                if start == 0:
                    action, reason = SENSOR_FAILURE, "gap at start of trace; no prior value to carry forward"
                elif not np.isfinite(duration):
                    action, reason = SENSOR_FAILURE, "gap duration unknown (no time_col or sample_interval_seconds)"
                elif duration <= max_gap_seconds:
                    action, reason = FORWARD_FILLED, f"gap of {duration:.3f}s within {max_gap_seconds}s tolerance"
                else:
                    action, reason = SENSOR_FAILURE, f"gap of {duration:.3f}s exceeds {max_gap_seconds}s tolerance"

                if action == FORWARD_FILLED:
                    df.loc[run_index, column] = df.at[index[start - 1], column]
                    df.loc[run_index, f"{column}_imputed"] = True
                else:
                    df.loc[run_index, f"{column}_sensor_failure"] = True

                report.events.append(GapEvent(
                    column=column, group=tuple(group_id), n_samples=n_samples,
                    duration_seconds=duration, action=action, reason=reason,
                ))

    report.completeness_after = _completeness(df, present)
    return df, report
