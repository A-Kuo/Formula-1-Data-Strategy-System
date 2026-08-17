# Feature Engineering

The authoritative source is `f1_pit_window.features.build_features` — this
document is a reading guide to it, not a duplicate of it. Every feature has
a versioned `MetricDefinition` (key, version, unit, prose definition,
rationale, effective-date range, pandas implementation, and — where
applicable — a SQL implementation tested against the pandas one). Run
`CANONICAL_REGISTRY.to_markdown()` for the always-current, generated table.

## Why a registry, not a feature-engineering script

The prior repo computed `DegradationRate` **four separate times**, once each
in `pipeline.py`, `feature_engineering.py`, `feature_engineering_real.py`,
and `load_real_data.py` — independently, never reconciled, never tested
against each other. `docs/decisions/004-canonical-metric-schema.md` has the
full account. A registry with exactly one governed definition per metric
per version is the direct fix, not a stylistic preference.

## The six features

| Feature | Unit | What it captures |
|---|---|---|
| `degradation_rate` | s/lap | Stint-level OLS slope of lap time vs. tyre age, clipped to ±0.5 s/lap |
| `stint_age_squared` | laps² | Non-linear tyre-age proxy: `tyre_age²` |
| `race_progress` | 0–1 | Fraction of the race elapsed |
| `pace_delta` | seconds | Lap time minus the driver's own trailing 5-lap median (self-relative) |
| `gap_closing_rate` | s/lap | Trailing-3-lap average change in instantaneous gap to leader; + = closing |
| `relative_pace_delta` | seconds | Lap time minus the field's median lap time at this lap (field-relative) |

The first four were the prior repo's model features, computed inline and
independently of any metric registry; they're governed here for the first
time (`docs/decisions/004-canonical-metric-schema.md`). The last two are new
— see "Strategy-context features" below.

## Strategy-context features: what got built, and what didn't

The prior repo's error analysis identified a real gap: the model sees tire
wear but no tactical context, so false positives cluster around drivers who
stay out on worn tires for strategic reasons (undercut/overcut), and false
negatives cluster around drivers who pit on fresh tires for the same reason.
Four features were proposed to address this. Three are implemented:

- **`gap_closing_rate`** — is this driver closing on the car ahead (a reason
  to stay out) or losing ground (a reason to pit)?
- **`relative_pace_delta`** — is this driver quick *for this point in the
  race*, not just quick for themselves?
- **`pit_lane_loss_by_circuit`** (`build_circuit_pit_loss_table` +
  `pit_lane_loss_by_circuit`) — a circuit's typical pit-stop cost, looked up
  from a table fit on historical data. Deliberately **not** in the versioned
  registry: unlike the other five features, it requires a fitted artifact
  (the per-circuit lookup table), the same reason a `StandardScaler` is fit
  on train and only ever applied to test. Folding it into the registry's
  `compute(frame)` contract — which assumes no external state — would risk
  a circuit's *own current lap* leaking into its own prediction.

The fourth, **`opponent_pit_window_signal`**, is not implemented.
`build_features.opponent_pit_window_signal()` raises `NotImplementedError`
with a docstring explaining why: it requires joint state across every driver
on track at a given lap (has anyone nearby pitted or is about to), which
means either a second pass after this model's own predictions exist for the
whole grid, or a materially different model architecture. That's a modeling
decision, not a missing afternoon of feature engineering — see
`docs/decisions/005-strategy-context-features.md`.

## Generating the SQL view

```bash
make view          # regenerate db/metrics_view.sql from the registry
make view-check     # CI gate: fails if the committed file has drifted
```

Only the four original metrics are SQL-backed
(`gap_to_leader`, `tyre_age`, `pit_delta`, `degradation_rate`) — the two
strategy-context features and the three formerly-ML-only ones are
pandas-only (`sql=None`), computed in the feature-build step before
training rather than exposed as a live view column. `build_metrics_view_sql`
excludes them from the generated SQL silently (logged, not raised); it only
raises if a metric with **no** SQL implementation is asked to appear in the
view for a date range where it's the canonical (live) version.
