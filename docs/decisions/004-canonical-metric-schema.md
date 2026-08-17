# 004 — Canonical metric schema, and the bugs building it caught

## Context

The prior repo's `DegradationRate` was reimplemented four separate times
across `pipeline.py`, `feature_engineering.py`, `feature_engineering_real.py`,
and `load_real_data.py` — never reconciled, never tested against each
other. `GapToLeader`, `GapToCarInFront`, and `PitDeltaEstimated` in two of
those files weren't even computed: `GapToCarInFront = 0.5` and
`PitDeltaEstimated = 25.4` were literal constants applied to every row.

## Decision

`f1_pit_window.features.build_features.CANONICAL_REGISTRY`: one governed
definition per metric per version, each with prose, rationale, an
effective-date range, a pandas implementation, and — where applicable — a
SQL implementation tested against the pandas one
(`tests/test_features_sql_parity.py`).

## What building it actually caught — not just what it fixed

Squashed history erases the wrong first drafts. Recorded here because
they're the actual demonstration of why a registry with tests beats an
inline reimplementation, not an abstract claim:

**`pit_delta@v2`'s first draft did a same-row subtraction**
(`pit_out_time - pit_in_time`), always null, because FastF1 records those
two timestamps on **different lap rows** (the in-lap and the following
out-lap). Caught by design review while building the test fixture — reading
FastF1's actual column semantics before writing synthetic data surfaced the
bug before a test ever ran. The fix uses `groupby().shift(1)` to pull the
previous lap's `pit_in_time` forward to the out-lap's `pit_out_time`. The
matching SQL uses `LAG(pit_in_time) OVER (... ORDER BY lap_number)` for the
same reason.

**`tyre_age_resets_after_pit`'s first draft kept off `is_pit_lap`**, which
`f1_pit_window.data.cleaning.normalize_laps` sets `True` on **both** the
in-lap and the out-lap of a stop — so the check demanded `tyre_age == 0` one
lap too late and failed on every single pit stop in a clean fixture. Caught
by `tests/test_validation.py::test_no_spurious_errors_on_well_formed_data`
(originally written specifically to catch a validation check crying wolf on
good data). Fixed by keying off a `stint_number` transition instead, which
changes exactly once per pit stop rather than twice.

**`compute_metric()` and `compute_all()` used two different column-naming
conventions** — versioned (`"gap_to_leader@v2"`) vs. bare
(`"gap_to_leader"`) — surfaced by a `KeyError` in an ad-hoc smoke test
before either convention had real callers depending on it. The decision:
`compute_all()`'s bare-key convention is correct, because a dashboard
querying "gap_to_leader" shouldn't need to know which version is currently
canonical to find its column — the entire point of a registry with exactly
one current version per key. `f1_pit_window.data.validation` was written to
match that convention.

## Why SQL/pandas parity uses a SQLite stand-in

The generated SQL targets PostgreSQL (`regr_slope(...) OVER (...)` for
`degradation_rate`, a native Postgres window aggregate). No PostgreSQL
instance was reachable in the sandbox this registry was built in
(`docker info` — no daemon access). Rather than skip the parity check or
mark it "manually verified," `tests/test_features_sql_parity.py` registers
a `regr_slope` window aggregate on SQLite via
`sqlite3.Connection.create_window_function` (Python 3.11+), implementing
the same incremental least-squares slope, plus `GREATEST`/`LEAST` scalar
shims SQLite also lacks — and actually executes the generated SQL against
it. A separate `test_parity_against_live_postgres`, skipped unless
`DATABASE_URL` points at a real Postgres, exists for CI (which does have a
Postgres service container) to exercise the real dialect, not just the
stand-in, forever.
