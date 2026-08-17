# Data Quality

Condensed from the prior repo's `DATA_INSPECTION_REPORT.md` (a 3-race,
3,860-lap manual inspection of Monaco/Monza/Singapore 2024) — the findings
below are what carried forward into this project's cleaning rules
(`f1_pit_window.data.cleaning`); the full original inspection notebook-style
report is not reproduced here to keep this document a reference, not a log.

## Raw shape

FastF1's `session.laps` returns one row per (driver, lap) with FastF1-native
column names and dtypes: `LapTime` as `timedelta64[ns]`, `Compound` as a raw
string whose vocabulary changed between the 2018 five-tier naming
(HYPERSOFT…SUPERHARD) and 2019+'s three-tier SOFT/MEDIUM/HARD, `DriverNumber`
as a string, `Driver` as a three-letter code. None of this matches the
canonical contract (`f1_pit_window.data.contracts`) without normalization.

## Missing values — expected vs. a real gap

`PitInTime`/`PitOutTime` are ~97% null **by construction** — only pit laps
have them, and that's correct, not a data quality defect. The distinction
between "null because this column doesn't apply to this row" and "null
because a sensor failed" matters: `f1_pit_window.data.cleaning.impute_gaps`
only ever acts on the second case, using a bounded forward-fill (see its
docstring), and never on a column like `pit_in_time` where null is the
expected value for most rows.

## Exclusion criteria and retention

Laps are excluded from training if any of:

| Condition | Why |
|---|---|
| `is_pit_lap` | Outcome variable — including it is direct label leakage |
| `track_status != 1` (safety car / VSC / red flag) | Different strategy regime; pit windows during a caution period aren't governed by tire degradation the way green-flag laps are |
| `lap_number <= 3` | Standing-start / formation-lap noise — unrepresentative lap times |
| `rainfall == True` | Distinct tire physics; wet-compound strategy isn't this model's target |

The prior repo's 3-race manual inspection measured **~78–87% retention**
after these exclusions, depending on the race's caution-period frequency.
The exact retention rate for the full 2018–2024 ingest depends on how many
seasons/races are actually pulled — re-run `scripts/ingest_data.py`, then
call `f1_pit_window.data.validation.validate_laps()` on the result (see its
`ValidationReport.summary()`) to get a current number rather than trusting a
3-race sample as representative of the full dataset. No CLI wraps this yet
— see `docs/decisions/` roadmap notes for adding one.

## Validation gate

Every batch — real or synthetic — passes through
`f1_pit_window.data.validation.validate_laps` before it's eligible for
publication (`f1_pit_window.data.repository.publish`): completeness (row
counts, null rates, driver coverage), consistency (monotonic cumulative gap,
tyre age resetting on stint change, strictly increasing lap numbers), and
reasonableness (lap-time bounds, degradation-rate bounds, an implausible
2-minute lap flagged rather than silently trained on). A failing batch is
rejected with a full audit trail; the last known-good snapshot keeps
serving. See `docs/decisions/` for two real bugs this layer's own test
suite caught during development.
