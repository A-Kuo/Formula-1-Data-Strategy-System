# F1 Pit-Window Decision Support

A reproducible machine-learning system that estimates whether a Formula 1
driver is likely to pit within the next 5 laps, using historical FastF1
race-session data.

It is a **decision-support prototype**, not an autonomous race-strategy
optimizer. The app surfaces uncertainty, threshold tradeoffs, and the
tactical context the current model does not observe — see
[`docs/limitations.md`](docs/limitations.md) before quoting a metric from
this project anywhere.

## What it demonstrates

- **Time-aware validation** — train on 2018–2023, evaluate once on a fully
  held-out 2024 season, never a random shuffle-split
  ([`docs/decisions/002-temporal-validation.md`](docs/decisions/002-temporal-validation.md)).
- **A canonical, versioned feature registry** — six governed metrics
  (`gap_to_leader`, `tyre_age`, `pit_delta`, `degradation_rate`,
  `stint_age_squared`, `race_progress`, `pace_delta`, plus two
  strategy-context features), each with a pandas implementation and — where
  applicable — a SQL implementation tested against it, replacing four
  independent, undocumented reimplementations of the same metric that
  existed in this project's history
  ([`docs/decisions/004-canonical-metric-schema.md`](docs/decisions/004-canonical-metric-schema.md)).
- **Pre-serving validation gates** — a batch of laps or a set of model
  artifacts is checked (completeness, consistency, physical plausibility;
  NaN/Inf, out-of-range probabilities, impossible saved metrics) before it
  reaches the dashboard. A failing batch is rejected with an audit trail;
  the last known-good snapshot keeps serving.
- **Calibration and group-wise evaluation** — not just an aggregate
  ROC-AUC. Brier score, a reliability curve, and metrics sliced by circuit
  or season, because a single number across an entire held-out season hides
  exactly the failure a reviewer is likely to ask about.
- **An honest engineering record** — [`docs/decisions/`](docs/decisions/)
  documents real bugs this project's own tests caught during development
  (a same-row timestamp subtraction that was always null, a validation
  check that flagged every pit stop as invalid, a feature whose first draft
  measured the wrong thing), not just the corrected final state.
- **A decision-workflow dashboard, not a tab bar** — three views (Race
  State, Decision Policy, Model Review) organized around what an analyst
  actually needs to do, not five loosely-related charts.

## Architecture

```
raw FastF1 laps
      │
      ▼
ingestion/fastf1_client.py   (fetch only — no normalization)
      │
      ▼
data/cleaning.py             (normalize + bounded sensor-gap imputation)
      │
      ▼
data/validation.py           (completeness / consistency / reasonableness)
      │
      ▼
data/repository.py           (publish-if-valid; else serve last known-good)
      │
      ▼
features/build_features.py   (versioned, SQL-tested metric registry)
      │
      ▼
modeling/train.py            (temporal split → LR/RF/XGBoost CV → threshold tuning)
      │
      ▼
modeling/{calibration,evaluate,inference}.py
      │
      ▼
app/streamlit_app.py         (Race State / Decision Policy / Model Review)
```

`monitoring/drift.py` (Population Stability Index) exists alongside this
pipeline but isn't wired to a live feed yet — see
[`docs/limitations.md`](docs/limitations.md).

## Status

This branch is a **structural scaffold**: the full pipeline is implemented
and tested (123 tests, 92%+ coverage on business logic — `make test`), but
**no data has been ingested and no model has been trained in this
environment**. `models/` and `results/` are intentionally empty. Run the
Quick Start below against a real `DATABASE_URL` to produce actual metrics —
see [`MIGRATION_MAP.md`](MIGRATION_MAP.md) for the full account of what's
ported, rewritten, new, and dropped from this project's prior iteration,
and what's still open.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Postgres (or point DATABASE_URL at your own instance)
make db-up

# 3. Ingest, apply the canonical metrics view, and train
make ingest
make apply-view
make train

# 4. Launch the dashboard
make app
```

Or the whole stack in containers: `make docker-up`.

## Development

```bash
make test     # full pytest suite, coverage-gated at 85%
make lint     # flake8 over src/ and tests/
make gate     # lint + SQL-drift check + tests — the CI gate, locally
make view     # regenerate db/metrics_view.sql after a feature-registry change
```

## Documentation

| Doc | Covers |
|---|---|
| [`docs/methodology.md`](docs/methodology.md) | Problem framing, temporal validation, target definition, model comparison |
| [`docs/data-quality.md`](docs/data-quality.md) | Exclusion criteria, retention, the validation gate |
| [`docs/feature-engineering.md`](docs/feature-engineering.md) | The six governed features, and what's deliberately not implemented |
| [`docs/model-evaluation.md`](docs/model-evaluation.md) | Metrics, calibration, group-wise evaluation, the metrics-discrepancy resolution |
| [`docs/limitations.md`](docs/limitations.md) | Scope, data-scope, and validation-scope limitations, stated up front |
| [`docs/decisions/`](docs/decisions/) | ADR-style records of specific design decisions and the bugs building them caught |
| [`MIGRATION_MAP.md`](MIGRATION_MAP.md) | What moved, what got rewritten, what got dropped from the prior iteration, and why |

## License

MIT License — see [`LICENSE`](LICENSE).
