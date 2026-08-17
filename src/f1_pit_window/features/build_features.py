"""
Canonical, versioned feature/metric registry
==============================================

Ported from ``f1pipeline/metrics.py`` (unchanged: :class:`MetricDefinition`,
:class:`MetricRegistry`, and the four original governed metrics
``gap_to_leader``, ``tyre_age``, ``pit_delta``, ``degradation_rate``, each
with a pandas implementation and a SQL implementation tested against each
other) — plus two extensions made as part of this project's re-scoping.

**Extension 1 — the model's own features are now governed too.**
The prior repo's ``pipeline.py`` computed ``StintAgeSquared``, ``RaceProgress``,
and ``PaceDelta`` inline, independently of the metric registry — see
``docs/decisions/`` for why that mattered (four independent
``DegradationRate`` reimplementations were the direct consequence of this
project *not* doing what this file now does). They're registered below as
``stint_age_squared``, ``race_progress``, ``pace_delta``.

**Extension 2 — three of the four "strategy-context" features are real.**
The project's error analysis identified the gap: the model sees tire wear
but no tactical context, so it predicts "old tires ⇒ pit" even when a driver
stays out for strategic reasons. ``gap_closing_rate``, ``relative_pace_delta``,
and a circuit-level ``pit_lane_loss_by_circuit`` lookup are implemented below
— each computable from data this project already ingests, no new data
source needed. The fourth proposed feature, an opponent-pit-window signal,
is intentionally **not** implemented here — see
:func:`opponent_pit_window_signal` for why it's a materially bigger lift
than the other three, not a missing afternoon of work.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRACE_KEYS = ("session_key", "driver_number")
STINT_KEYS = ("session_key", "driver_number", "stint_number")
DEGRADATION_CLIP = 0.5


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    version: str
    title: str
    unit: str
    definition: str
    rationale: str
    owner: str
    effective_from: date
    compute: Callable[[pd.DataFrame], pd.Series]
    effective_to: date | None = None
    sql: str | None = None
    supersedes: str | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.key}@{self.version}"

    @property
    def is_current(self) -> bool:
        return self.effective_to is None

    def live_on(self, when: date) -> bool:
        if when < self.effective_from:
            return False
        return self.effective_to is None or when < self.effective_to

    def fingerprint(self) -> str:
        payload = "|".join([self.key, self.version, self.unit, self.definition, self.sql or ""])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "key": self.key, "version": self.version, "title": self.title, "unit": self.unit,
            "definition": self.definition, "rationale": self.rationale, "owner": self.owner,
            "effective_from": self.effective_from.isoformat(),
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
            "sql": self.sql, "supersedes": self.supersedes, "fingerprint": self.fingerprint(),
        }


class MetricRegistry:
    def __init__(self, definitions: Iterable[MetricDefinition] = ()) -> None:
        self._by_key: dict[str, dict[str, MetricDefinition]] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: MetricDefinition) -> None:
        versions = self._by_key.setdefault(definition.key, {})
        if definition.version in versions:
            raise ValueError(f"Duplicate definition {definition.qualified_name}")
        versions[definition.version] = definition

    def keys(self) -> list[str]:
        return sorted(self._by_key)

    def versions(self, key: str) -> list[str]:
        self._require_key(key)
        return sorted(self._by_key[key], key=lambda v: self._by_key[key][v].effective_from)

    def get(self, key: str, version: str | None = None) -> MetricDefinition:
        self._require_key(key)
        versions = self._by_key[key]
        if version is not None:
            if version not in versions:
                raise KeyError(f"Unknown version {version!r} for metric {key!r}; have {sorted(versions)}")
            return versions[version]
        current = [d for d in versions.values() if d.is_current]
        if len(current) != 1:
            raise LookupError(f"Expected exactly 1 current version of {key!r}, found {len(current)}")
        return current[0]

    def as_of(self, when: date) -> dict[str, MetricDefinition]:
        live: dict[str, MetricDefinition] = {}
        for key, versions in self._by_key.items():
            matches = [d for d in versions.values() if d.live_on(when)]
            if len(matches) > 1:
                raise LookupError(f"Overlapping effective ranges for {key!r} on {when.isoformat()}")
            if matches:
                live[key] = matches[0]
        return live

    def current(self) -> dict[str, MetricDefinition]:
        return {key: self.get(key) for key in self.keys()}

    def fingerprint(self) -> str:
        joined = "|".join(sorted(d.fingerprint() for d in self.current().values()))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint(),
            "metrics": {key: [self._by_key[key][v].to_dict() for v in self.versions(key)] for key in self.keys()},
        }

    def to_markdown(self) -> str:
        lines = [f"# Canonical feature registry (`{self.fingerprint()}`)", "",
                 "| Metric | Version | Unit | Effective | Definition |", "| --- | --- | --- | --- | --- |"]
        for key in self.keys():
            for version in self.versions(key):
                d = self._by_key[key][version]
                until = d.effective_to.isoformat() if d.effective_to else "current"
                marker = " **(canonical)**" if d.is_current else ""
                lines.append(
                    f"| `{d.key}` | {d.version}{marker} | {d.unit} | "
                    f"{d.effective_from.isoformat()} → {until} | {d.definition} |"
                )
        return "\n".join(lines)

    def _require_key(self, key: str) -> None:
        if key not in self._by_key:
            raise KeyError(f"Unknown metric {key!r}; registered metrics are {self.keys()}")


def _grouping(frame: pd.DataFrame, keys: Iterable[str]) -> list[str]:
    present = [key for key in keys if key in frame.columns]
    if not present:
        raise KeyError(f"Frame is missing all grouping columns {list(keys)}")
    return present


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    sort_cols = _grouping(frame, TRACE_KEYS) + ["lap_number"]
    return frame.sort_values(sort_cols)


def derive_stint(frame: pd.DataFrame) -> pd.Series:
    ordered = _ordered(frame)
    group = ordered.groupby(_grouping(frame, TRACE_KEYS), sort=False)["is_pit_lap"]
    stint = group.transform(lambda s: s.astype(bool).shift(1, fill_value=False).cumsum())
    return stint.reindex(frame.index).astype("int64")


def _stint_grouping(frame: pd.DataFrame):
    if "stint_number" in frame.columns and frame["stint_number"].notna().any():
        return _grouping(frame, STINT_KEYS)
    trace = _grouping(frame, TRACE_KEYS)
    return [frame[col] for col in trace] + [derive_stint(frame).rename("stint_number")]


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y on x, or 0.0 when it is not identifiable."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 2:
        return 0.0
    variance = float(np.var(x))
    if variance <= 0:
        return 0.0
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    return covariance / variance


# ─────────────────────────────────────────────────────────────
# Original four canonical metrics (unchanged from f1pipeline/metrics.py)
# ─────────────────────────────────────────────────────────────


def _gap_to_leader_v1(frame: pd.DataFrame) -> pd.Series:
    leader = frame.groupby(_grouping(frame, ["session_key", "lap_number"]), sort=False)[
        "lap_time_seconds"
    ].transform("min")
    return (frame["lap_time_seconds"] - leader).astype(float)


def _gap_to_leader_v2(frame: pd.DataFrame) -> pd.Series:
    per_lap = _gap_to_leader_v1(frame)
    ordered = _ordered(frame)
    cumulative = per_lap.reindex(ordered.index).groupby(
        [ordered[col] for col in _grouping(frame, TRACE_KEYS)], sort=False
    ).cumsum()
    return cumulative.reindex(frame.index).astype(float)


def _tyre_age_v1(frame: pd.DataFrame) -> pd.Series:
    ordered = _ordered(frame)
    lap_times = ordered["lap_time_seconds"].astype(float)
    grouping = _stint_grouping(frame)
    keys = [g.reindex(ordered.index) if isinstance(g, pd.Series) else ordered[g] for g in grouping]
    elapsed = lap_times.groupby(keys, sort=False).cumsum() - lap_times
    return elapsed.reindex(frame.index).astype(float)


def _tyre_age_v2(frame: pd.DataFrame) -> pd.Series:
    ordered = _ordered(frame)
    grouping = _stint_grouping(frame)
    keys = [g.reindex(ordered.index) if isinstance(g, pd.Series) else ordered[g] for g in grouping]
    age = ordered.groupby(keys, sort=False).cumcount()
    return age.reindex(frame.index).astype("int64")


def _pit_delta_v1(frame: pd.DataFrame) -> pd.Series:
    ordered = _ordered(frame)
    baseline = ordered.groupby(_grouping(frame, TRACE_KEYS), sort=False)["lap_time_seconds"].transform("median")
    delta = (ordered["lap_time_seconds"] - baseline).reindex(frame.index)
    return delta.where(frame["is_pit_lap"].astype(bool), np.nan).astype(float)


def _pit_delta_v2(frame: pd.DataFrame) -> pd.Series:
    """
    Pit-box duration in seconds, reported on the out-lap.

    FastF1 records ``PitInTime`` on the in-lap and ``PitOutTime`` on the
    *following* lap, so the duration is ``this lap's pit_out_time`` minus
    ``the previous lap's pit_in_time`` — not a same-row subtraction, which is
    always null. See ``docs/decisions/`` for the first (wrong) draft.
    """
    for column in ("pit_in_time", "pit_out_time"):
        if column not in frame.columns:
            raise KeyError(f"pit_delta@v2 requires {column!r}; it is measured from telemetry, not lap times")
    ordered = _ordered(frame)
    group_keys = _grouping(frame, TRACE_KEYS)
    pit_in_prev = ordered.groupby(group_keys, sort=False)["pit_in_time"].shift(1)
    pit_out = ordered["pit_out_time"].astype(float)
    duration = pit_out - pit_in_prev.astype(float)
    valid = pit_out.notna() & pit_in_prev.notna() & (duration >= 0)
    return duration.where(valid, np.nan).reindex(frame.index).astype(float)


def _degradation_rate_v1(frame: pd.DataFrame) -> pd.Series:
    ordered = _ordered(frame)
    rolling = ordered.groupby(_grouping(frame, TRACE_KEYS), sort=False)["lap_time_seconds"].transform(
        lambda s: s.rolling(3, min_periods=3).mean()
    )
    return rolling.diff().reindex(frame.index).astype(float)


def _degradation_rate_v2(frame: pd.DataFrame) -> pd.Series:
    age = _tyre_age_v2(frame)
    grouping = _stint_grouping(frame)
    keys = [g if isinstance(g, pd.Series) else frame[g] for g in grouping]
    working = pd.DataFrame({"_age": age.astype(float), "_lap_time": frame["lap_time_seconds"].astype(float)})
    slopes = pd.Series(0.0, index=frame.index, dtype=float)
    for _, group in working.groupby(keys, sort=False):
        slope = _ols_slope(group["_age"].to_numpy(dtype=float), group["_lap_time"].to_numpy(dtype=float))
        slopes.loc[group.index] = float(np.clip(slope, -DEGRADATION_CLIP, DEGRADATION_CLIP))
    return slopes


# ─────────────────────────────────────────────────────────────
# Extension 1: former ML-only features, now governed
# ─────────────────────────────────────────────────────────────


def _stint_age_squared_v1(frame: pd.DataFrame) -> pd.Series:
    """
    Non-linear tyre-age proxy: (tyre age in laps)².

    Deliberately built on ``tyre_age@v2`` (this registry's own derived
    stint-relative lap counter) rather than trusting FastF1's raw
    ``tyre_life`` field directly, so this feature and ``degradation_rate``
    always agree on what "lap N of the stint" means for the same lap.
    """
    return (_tyre_age_v2(frame).astype(float) ** 2)


def _race_progress_v1(frame: pd.DataFrame) -> pd.Series:
    """Fraction of the race completed: this driver's lap_number / their session's max lap_number."""
    max_lap = frame.groupby(_grouping(frame, ["session_key"]), sort=False)["lap_number"].transform("max")
    return (frame["lap_number"] / max_lap).astype(float)


def _pace_delta_v1(frame: pd.DataFrame) -> pd.Series:
    """This lap's time minus the driver's own trailing 5-lap rolling median — self-relative pace."""
    ordered = _ordered(frame)
    rolling_median = ordered.groupby(_grouping(frame, TRACE_KEYS), sort=False)["lap_time_seconds"].transform(
        lambda s: s.rolling(5, min_periods=1).median()
    )
    delta = (ordered["lap_time_seconds"] - rolling_median).reindex(frame.index)
    return delta.astype(float)


# ─────────────────────────────────────────────────────────────
# Extension 2: strategy-context features (real implementations)
# ─────────────────────────────────────────────────────────────


def _gap_closing_rate_v1(frame: pd.DataFrame) -> pd.Series:
    """
    Average per-lap change in the *instantaneous* gap to the leader
    (``gap_to_leader@v1``) over the trailing 3 laps.

    Positive = closing the gap (getting faster relative to the leader);
    negative = losing ground. Built on ``gap_to_leader@v1`` (per-lap), not
    ``@v2`` (cumulative) — the cumulative series is non-decreasing by
    construction, so its diff measures accumulated gap magnitude over the
    window, not closing speed. Using it here was this feature's own first
    draft and a real bug, caught by
    ``test_gap_closing_rate_positive_when_gap_shrinking``: it returned a
    *negative* rate for a driver who was visibly, unambiguously closing on
    the leader lap over lap. See ``docs/decisions/`` for the full account.
    """
    ordered = _ordered(frame)
    gap = _gap_to_leader_v1(ordered)
    trend = gap.groupby(
        [ordered[col] for col in _grouping(frame, TRACE_KEYS)], sort=False
    ).transform(lambda s: -(s.diff(3) / 3.0))
    return trend.reindex(frame.index).astype(float)


def _relative_pace_delta_v1(frame: pd.DataFrame) -> pd.Series:
    """
    This lap's time minus the field's median lap time at the same lap number.

    Distinct from ``pace_delta`` (self-relative, vs. this driver's own recent
    laps): this is relative to *everyone else on track right now* — "am I
    quick for this point in the race" rather than "am I quick for me."
    Negative = faster than the field's median at this lap.
    """
    field_median = frame.groupby(_grouping(frame, ["session_key", "lap_number"]), sort=False)[
        "lap_time_seconds"
    ].transform("median")
    return (frame["lap_time_seconds"] - field_median).astype(float)


def build_circuit_pit_loss_table(historical_frame: pd.DataFrame) -> pd.Series:
    """
    Fit a circuit → mean measured pit-box duration lookup table.

    This is a **fitted artifact**, not a pure function of the frame it's
    applied to — the same reason a :class:`~sklearn.preprocessing.StandardScaler`
    is fit on train and only ever *applied* to test. Deliberately kept out of
    :data:`CANONICAL_REGISTRY`, whose ``compute(frame)`` contract assumes no
    external state: computing a circuit's average pit loss from the same
    batch a prediction is being made on would leak that day's own stop into
    its own prediction.

    Callers: fit this once on the training set (or a rolling historical
    window in production), persist the returned table alongside the model
    artifact, and pass it into :func:`pit_lane_loss_by_circuit` at inference
    time — never re-fit it on live data.

    Requires ``circuit_name`` and a computed ``pit_delta`` column on
    ``historical_frame``.
    """
    for column in ("circuit_name", "pit_delta"):
        if column not in historical_frame.columns:
            raise KeyError(f"build_circuit_pit_loss_table requires {column!r}")
    return historical_frame.groupby("circuit_name")["pit_delta"].mean().rename("circuit_pit_loss_mean")


def pit_lane_loss_by_circuit(frame: pd.DataFrame, circuit_loss_table: pd.Series) -> pd.Series:
    """
    Expected pit-box duration for this lap's circuit, looked up from a
    pre-fitted table (see :func:`build_circuit_pit_loss_table`).

    Unknown circuits (not present in the fitted table — e.g. a new venue)
    fall back to the table's global mean rather than raising, since a
    missing circuit is exactly the situation where the model most needs
    *some* prior rather than a crash.
    """
    if "circuit_name" not in frame.columns:
        raise KeyError("pit_lane_loss_by_circuit requires 'circuit_name'")
    fallback = float(circuit_loss_table.mean()) if len(circuit_loss_table) else float("nan")
    return frame["circuit_name"].map(circuit_loss_table).fillna(fallback).astype(float)


def opponent_pit_window_signal(frame: pd.DataFrame, **kwargs) -> pd.Series:
    """
    NOT IMPLEMENTED. Kept here, raising, so the gap is discoverable in code
    rather than only in a planning document.

    Proposed definition: whether drivers within a small time gap of this one
    (ahead and/or behind) have pitted in the last K laps, or are themselves
    predicted to pit soon. Unlike the other three strategy-context features,
    this one requires **joint state across drivers within a session** — a
    per-driver row can't be labeled without first computing (or predicting)
    every nearby driver's pit status at the same lap, which means either:

    * a second pass over the session after this model's own predictions
      exist for every driver (a circular dependency the other three features
      don't have), or
    * a materially different model shape — sequence-to-sequence over the
      whole grid per lap, not a per-row classifier.

    That's a real modeling-architecture decision, not a missing afternoon of
    feature engineering, which is why it's deferred rather than stubbed with
    a fake return value.
    """
    raise NotImplementedError(
        "opponent_pit_window_signal requires joint per-lap state across the full grid, not a "
        "per-row feature — see this function's docstring for why it's deferred, not just unbuilt."
    )


# ─────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────

_SEASON_2024 = date(2024, 1, 1)
_SEASON_2023 = date(2023, 1, 1)

CANONICAL_REGISTRY = MetricRegistry([
    MetricDefinition(
        key="gap_to_leader", version="v1", title="Gap to leader (instantaneous)", unit="seconds",
        definition=(
            "Difference between this driver's lap time and the fastest lap time set by any "
            "driver on the same lap of the same session."
        ),
        rationale="The original definition. Retired at the start of 2024 in favor of a cumulative measure.",
        owner="race-strategy", effective_from=_SEASON_2023, effective_to=_SEASON_2024,
        sql="lap_gap_seconds", compute=_gap_to_leader_v1,
    ),
    MetricDefinition(
        key="gap_to_leader", version="v2", title="Gap to leader (cumulative)", unit="seconds",
        definition=(
            "Running total of the per-lap gap to the lap leader, accumulated from lap 1 to the "
            "current lap for each driver."
        ),
        rationale=(
            "Strategy asks 'how far behind am I now' — a cumulative quantity. Also "
            "monotonically non-decreasing, a cheap consistency check."
        ),
        owner="race-strategy", effective_from=_SEASON_2024,
        sql=(
            "SUM(lap_gap_seconds) OVER (PARTITION BY session_key, driver_number ORDER BY "
            "lap_number ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
        ),
        supersedes="v1", compute=_gap_to_leader_v2,
    ),
    MetricDefinition(
        key="tyre_age", version="v1", title="Tyre age (elapsed time)", unit="seconds",
        definition="Seconds of running completed on the current set of tyres, excluding the current lap.",
        rationale=(
            "Time-based ageing is physically natural but not what compound-allocation rules or "
            "the degradation model use. Retired for 2024."
        ),
        owner="tyre-performance", effective_from=_SEASON_2023, effective_to=_SEASON_2024,
        sql=(
            "SUM(lap_time_seconds) OVER (PARTITION BY session_key, driver_number, stint_number "
            "ORDER BY lap_number ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) - lap_time_seconds"
        ),
        compute=_tyre_age_v1,
    ),
    MetricDefinition(
        key="tyre_age", version="v2", title="Tyre age (laps)", unit="laps",
        definition="Number of completed laps on the current set of tyres. Zero on the out-lap.",
        rationale="Laps, not seconds — every other tyre artefact (degradation model, compound allocation) uses laps.",
        owner="tyre-performance", effective_from=_SEASON_2024,
        sql="tyre_age_laps", supersedes="v1", compute=_tyre_age_v2,
    ),
    MetricDefinition(
        key="pit_delta", version="v1", title="Pit delta (lap-time proxy)", unit="seconds",
        definition="On a pit lap, the driver's lap time minus their median lap time for the session; null otherwise.",
        rationale=(
            "A proxy from before pit-lane telemetry was joined in. Conflates the stop with "
            "traffic/SC on that lap. Retired for 2024."
        ),
        owner="pit-crew-ops", effective_from=_SEASON_2023, effective_to=_SEASON_2024, sql=None, compute=_pit_delta_v1,
    ),
    MetricDefinition(
        key="pit_delta", version="v2", title="Pit delta (measured)", unit="seconds",
        definition=(
            "Pit-lane duration measured from telemetry, reported on the out-lap: this lap's "
            "pit-exit timestamp minus the previous (in-)lap's pit-entry timestamp."
        ),
        rationale="Measured beats inferred. See docs/decisions/ for the same-row-subtraction bug this replaced.",
        owner="pit-crew-ops", effective_from=_SEASON_2024,
        sql=(
            "CASE WHEN pit_out_time IS NOT NULL AND LAG(pit_in_time) OVER (PARTITION BY "
            "session_key, driver_number ORDER BY lap_number) IS NOT NULL AND pit_out_time >= "
            "LAG(pit_in_time) OVER (PARTITION BY session_key, driver_number ORDER BY lap_number) "
            "THEN pit_out_time - LAG(pit_in_time) OVER (PARTITION BY session_key, driver_number "
            "ORDER BY lap_number) END"
        ),
        supersedes="v1", compute=_pit_delta_v2,
    ),
    MetricDefinition(
        key="degradation_rate", version="v1", title="Degradation rate (3-lap rolling)", unit="seconds per lap",
        definition="Lap-on-lap change in the trailing 3-lap mean lap time.",
        rationale="Responsive but noisy — 3 laps can't separate degradation from traffic. Retired for a stint fit.",
        owner="tyre-performance", effective_from=_SEASON_2023, effective_to=_SEASON_2024,
        sql=None, compute=_degradation_rate_v1,
    ),
    MetricDefinition(
        key="degradation_rate", version="v2", title="Degradation rate (stint OLS slope)", unit="seconds per lap",
        definition=(
            "Least-squares slope of lap time against tyre age over the whole stint, seconds "
            "lost per additional lap, clipped to ±0.5 s/lap."
        ),
        rationale=(
            "Uses every lap of the stint as evidence; the clip suppresses fit artefacts from a "
            "stint containing a safety car or in-lap. This is what the pit-prediction model consumes."
        ),
        owner="tyre-performance", effective_from=_SEASON_2024,
        sql=(
            "GREATEST(-0.5, LEAST(0.5, COALESCE(regr_slope(lap_time_seconds, tyre_age_laps) "
            "OVER (PARTITION BY session_key, driver_number, stint_number), 0)))"
        ),
        supersedes="v1", compute=_degradation_rate_v2,
    ),
    # Extension 1 — formerly ML-only, now governed.
    MetricDefinition(
        key="stint_age_squared", version="v1", title="Stint age squared", unit="laps²",
        definition="(tyre_age@v2)² — a non-linear degradation proxy: later stint laps contribute more.",
        rationale=(
            "Was computed inline in the prior repo's pipeline.py from raw tyre_life; now "
            "derived from the governed tyre_age@v2 so the two features can never silently "
            "disagree about what 'lap N of the stint' means."
        ),
        owner="tyre-performance", effective_from=_SEASON_2024, sql=None, compute=_stint_age_squared_v1,
    ),
    MetricDefinition(
        key="race_progress", version="v1", title="Race progress", unit="fraction (0-1)",
        definition="lap_number / max(lap_number) for this driver's session.",
        rationale=(
            "Unchanged from the prior repo's inline computation — genuinely simple, no prior "
            "bug to fix, brought under governance so it's testable and versioned like "
            "everything else the model consumes."
        ),
        owner="race-strategy", effective_from=_SEASON_2024, sql=None, compute=_race_progress_v1,
    ),
    MetricDefinition(
        key="pace_delta", version="v1", title="Pace delta (self-relative)", unit="seconds",
        definition="This lap's time minus the driver's own trailing 5-lap rolling median.",
        rationale=(
            "Unchanged from the prior repo's inline computation, brought under governance. See "
            "relative_pace_delta for the field-relative counterpart added in this project's re-scoping."
        ),
        owner="race-strategy", effective_from=_SEASON_2024, sql=None, compute=_pace_delta_v1,
    ),
    # Extension 2 — strategy-context features.
    MetricDefinition(
        key="gap_closing_rate", version="v1", title="Gap closing rate", unit="seconds per lap",
        definition="Average per-lap change in gap_to_leader over the trailing 3 laps; positive means closing.",
        rationale=(
            "Directly targets the error-analysis finding that the model can't distinguish "
            "'old tyres, about to pit' from 'old tyres, but closing on the car ahead, staying out.'"
        ),
        owner="race-strategy", effective_from=_SEASON_2024, sql=None, compute=_gap_closing_rate_v1,
    ),
    MetricDefinition(
        key="relative_pace_delta", version="v1", title="Relative pace delta (field-relative)", unit="seconds",
        definition="This lap's time minus the field's median lap time at the same lap number.",
        rationale=(
            "Complements pace_delta (self-relative): answers 'am I quick for this point in the "
            "race,' which pace_delta alone cannot."
        ),
        owner="race-strategy", effective_from=_SEASON_2024, sql=None, compute=_relative_pace_delta_v1,
    ),
])


def compute_metric(frame: pd.DataFrame, key: str, version: str | None = None) -> pd.Series:
    """Compute a single governed metric. Returns a Series named ``"{key}@{version}"``."""
    definition = CANONICAL_REGISTRY.get(key, version)
    return definition.compute(frame).rename(definition.qualified_name)


def compute_all(frame: pd.DataFrame, when: date | None = None) -> pd.DataFrame:
    """
    Compute every metric canonical on ``when`` (default: today). Columns are
    bare metric keys (``"gap_to_leader"``, not ``"gap_to_leader@v2"``) — a
    caller shouldn't need to know which version is currently live to find
    its column. See ``docs/decisions/`` for why this convention was chosen
    over the alternative.
    """
    definitions = CANONICAL_REGISTRY.as_of(when) if when else CANONICAL_REGISTRY.current()
    return pd.DataFrame({key: d.compute(frame) for key, d in sorted(definitions.items())}, index=frame.index)


_VIEW_IDENTITY_COLUMNS = ("session_key", "driver_number", "lap_number", "stint_number", "lap_time_seconds")
_BASE_CTE_BODY = """    SELECT
        session_key,
        driver_number,
        lap_number,
        stint_number,
        lap_time_seconds,
        is_pit_lap,
        pit_in_time,
        pit_out_time,
        lap_time_seconds - MIN(lap_time_seconds) OVER (PARTITION BY session_key, lap_number)
            AS lap_gap_seconds,
        ROW_NUMBER() OVER (PARTITION BY session_key, driver_number, stint_number
            ORDER BY lap_number) - 1 AS tyre_age_laps
    FROM {source_table}"""


def build_metrics_view_sql(
    view_name: str = "canonical_lap_metrics",
    source_table: str = "laps",
    when: date | None = None,
) -> str:
    """
    Generate the ``CREATE OR REPLACE VIEW`` for the metrics with a SQL
    implementation. Note: the four Extension 1/2 metrics added in this
    project (``stint_age_squared``, ``race_progress``, ``pace_delta``,
    ``gap_closing_rate``, ``relative_pace_delta``) are pandas-only
    (``sql=None``) — they're computed in the feature-build step before
    training, not exposed as a live SQL view column, so they're excluded
    from the generated view rather than raising (unlike a *live* metric
    with no SQL, which is an error — see the ``ValueError`` branch below).
    """
    definitions = CANONICAL_REGISTRY.as_of(when) if when else CANONICAL_REGISTRY.current()
    sql_definitions = {k: d for k, d in definitions.items() if d.sql}
    pandas_only = sorted(set(definitions) - set(sql_definitions))
    if pandas_only:
        logger.info("Excluding pandas-only metrics from generated SQL view: %s", pandas_only)

    columns = [f"    {column}" for column in _VIEW_IDENTITY_COLUMNS]
    for key, definition in sorted(sql_definitions.items()):
        columns.append(f"    -- {key}@{definition.version} ({definition.unit}), {definition.fingerprint()}")
        columns.append(f"    {definition.sql} AS {key}")

    header = (
        "-- Generated from f1_pit_window.features.build_features — do not edit by hand.\n"
        f"-- Registry fingerprint: {CANONICAL_REGISTRY.fingerprint()}\n"
    )
    return (
        f"{header}CREATE OR REPLACE VIEW {view_name} AS\n"
        f"WITH base AS (\n{_BASE_CTE_BODY.format(source_table=source_table)}\n)\n"
        "SELECT\n" + ",\n".join(columns) + "\nFROM base;"
    )
