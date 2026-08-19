# Migration Map: what moved, what got rewritten, what got dropped

This is the record of the move from the prior repo
(`A-Kuo/FastF1-Pit-Strategy-Optimization`) into this one. That repo staged the
rewrite on a `new-repo-scaffold` branch; this repo is where it now lives, and
the prior repo is superseded. Everything below is categorized honestly: **ported** (identical logic, new
location/imports), **rewritten** (same intent, materially different
implementation), **new** (didn't exist before), or **dropped** (not carried
forward, with the reason).

## Ported — working code, moved and renamed, logic unchanged

| Old location | New location | Notes |
|---|---|---|
| `f1pipeline/normalize.py` | `src/f1_pit_window/data/contracts.py` + `.../cleaning.py` | Split: schema constants → `contracts.py`, transform functions → `cleaning.py` |
| `f1pipeline/imputation.py` | `src/f1_pit_window/data/cleaning.py` | Merged into cleaning — normalize and impute are one pipeline stage in the target architecture |
| `f1pipeline/validation.py` | `src/f1_pit_window/data/validation.py` | Unchanged logic (already includes the `tyre_age_resets_after_pit` stint-based fix) |
| `f1pipeline/gate.py` | `src/f1_pit_window/data/repository.py` | Renamed to match "repository" as the persistence boundary |
| `f1pipeline/metrics.py` | `src/f1_pit_window/features/build_features.py` | The four original metrics are unchanged; see "Rewritten" for what's new in this file |
| `f1pipeline/model_metrics.py` | Split: `validate_model_artifacts`/`SavedMetrics` → `modeling/inference.py`; `threshold_sweep` → `modeling/evaluate.py` | Split by concern: pre-serving gate vs. evaluation |
| `tests/test_normalize.py` + `test_imputation.py` | `tests/test_cleaning.py` | Merged to match the module merge |
| `tests/test_metrics.py` | `tests/test_build_features.py` | Extended with governance tests for the new registry entries |
| `tests/test_metrics_sql_parity.py` | `tests/test_features_sql_parity.py` | Unchanged approach (SQLite `regr_slope` stand-in); now covers 4 metrics, same as before |
| `tests/test_validation.py`, `test_gate.py` | `tests/test_validation.py`, `test_repository.py` | Renamed to match; `test_validation.py` gained a regression test for the stint-based fix |
| `tests/test_model_metrics.py` | Split: `test_inference.py`, part of `test_evaluate.py` | Matches the module split |
| `.github/workflows/ci.yml` | Same path | Paths updated for `src/` layout; otherwise unchanged (Postgres service container, lint, SQL-drift check, pytest) |
| `scripts/generate_metrics_view.py` | Same path | Import path updated for `src/f1_pit_window` |

## Rewritten — same intent, materially different implementation

| What | Why it's not a straight port |
|---|---|
| `scripts/ingest.py` → `src/f1_pit_window/ingestion/fastf1_client.py` + `scripts/ingest_data.py` | Split fetch (no normalization) from clean (normalize + write) — this is the architectural fix for the `Time`/`LapTime` column bug (`docs/decisions/004-canonical-metric-schema.md`): fetch and normalize used to be one function, so there was no seam to test normalization independent of hitting the live FastF1 API. |
| `pipeline.py` → `src/f1_pit_window/modeling/train.py` + `scripts/train_model.py` | No longer generates synthetic data inline. Sources features from `build_features.py` instead of an inline `ols_slope()` reimplementation — this is what actually resolves the four-duplicate-`DegradationRate` problem, not just documents it. Synthetic data generation itself is **not ported** — see Dropped. |
| `streamlit_app_enhanced.py` → `src/f1_pit_window/app/streamlit_app.py` | Restructured from 5 loosely-related tabs into 3 views organized around a decision workflow (Race State / Decision Policy / Model Review), per the re-scoping plan. Calls `calibration.py`/`evaluate.py`/`inference.py` instead of computing anything inline. |
| `scripts/ingest.py`'s `RaceORM`/`LapORM` | `src/f1_pit_window/data/schema.py` | Consolidated with `sql_utils.py`'s *competing, incompatible* ORM (see Dropped) onto one schema — the prior repo had two independently-defined `laps` tables that were never reconciled, the same failure mode as the duplicate `DegradationRate` implementations, one layer down. |

## New — did not exist in the prior repo

| File | What it adds |
|---|---|
| `src/f1_pit_window/modeling/calibration.py` | Brier score + reliability curve. The prior repo displayed a "pit probability" with no check that the probability values were calibrated. |
| `src/f1_pit_window/modeling/evaluate.py`'s `group_wise_metrics` | Metrics sliced by circuit/season/compound/etc., not just one aggregate number across the whole held-out season. |
| `src/f1_pit_window/monitoring/drift.py` | Population Stability Index feature-drift detection. Detection only — no automated retraining trigger (a real operational decision, deliberately deferred, not hardcoded — see `docs/limitations.md`). |
| `build_features.py`'s `stint_age_squared`, `race_progress`, `pace_delta` | The prior repo's model features, computed inline four times over; now single, governed, versioned definitions. |
| `build_features.py`'s `gap_closing_rate`, `relative_pace_delta`, `pit_lane_loss_by_circuit` | Three of the four proposed strategy-context features, built for real (see `docs/decisions/005-strategy-context-features.md`). |
| `scripts/train_model.py`, `scripts/ingest_data.py`, `scripts/apply_metrics_view.py` | Actual runnable entry points tying the package together — the prior repo's scripts existed but several had real bugs in their invocation (see `docs/decisions/`). |
| `src/f1_pit_window/monitoring/benchmark.py`, `scripts/run_benchmark.py`, `metrics/benchmark_results.csv` | `make benchmark` times the load/feature-build/train stages and appends runtime + row-throughput, tagged with the producing commit, to a git-tracked CSV — instrumented pipeline performance tracking the prior repo had no equivalent of. |
| `scripts/validate_artifacts.py`, `scripts/format_release_notes.py`, `.github/workflows/scheduled-retrain.yml` | A cron-triggered (~every 4 months) workflow that re-ingests new F1 seasons, retrains, gates the result through the existing `validate_model_artifacts` check before publishing anything, then releases the model as a dated GitHub Release and commits a fresh `metrics/benchmark_results.csv` row — the prior repo had no scheduled refresh of any kind. |
| `Makefile` | `make test`, `make lint`, `make gate` (the CI gate, locally), `make ingest`, `make train`, `make benchmark`, `make app`, `make docker-up`. |
| `docs/` (methodology, data-quality, feature-engineering, model-evaluation, limitations, decisions/001–007) | Replaces the root-level phase-summary sprawl (`FINAL_SUMMARY.md`, `TECHNICAL_SUMMARY.md`, `TASK_COMPLETION_SUMMARY.md`, `PHASE_1_*.md`, `QUICK_REFERENCE.md`) with one navigable structure. `007` also adds a disclaimer header to `research/*.md`'s speculative FastAPI/Kafka brainstorming notes, pointing back at the actual (monolith) architecture. |
| `pyproject.toml`, `setup.cfg` | pytest config (`pythonpath = ["src"]`, coverage settings) and flake8 config, replacing a bare `pytest.ini`. |

## Dropped — not carried forward, with the reason

| What | Why |
|---|---|
| `streamlit_app.py` (the 14-feature dashboard) | Dead code — its required model artifacts (`models/random_forest.pkl`, etc.) never existed in the committed repo. See `docs/decisions/006-dual-pipeline-discovery.md`. |
| `load_real_data.py`, `feature_engineering_real.py`, `feature_engineering.py`, `model_comparison.py`, `model_comparison_enhanced.py` | The pipelines that fed the dead dashboard above, plus the fabricated `GapToLeader`/`GapToCarInFront`/`PitDeltaEstimated` constants they computed. Their real logic (the `DegradationRate` OLS slope, the target-window computation) is superseded by `build_features.py` and `train.py`. |
| `sql_utils.py` | A second, incompatible `RaceORM`/`LapORM` schema (baked-in feature columns instead of raw telemetry + a queryable view) that was never reconciled with `scripts/ingest.py`'s schema. `src/f1_pit_window/data/schema.py` consolidates on the raw-telemetry shape. |
| `config.py`, `logging_config.py` | Multi-database (PostgreSQL/MySQL/SQL Server) configuration abstraction, unused by anything in `src/f1_pit_window`, which talks to Postgres via a single `DATABASE_URL` env var — consistent with the re-scoping away from "five projects at once." |
| `data_inspection.py` | One-off exploratory script. Its substantive findings are carried forward into `docs/data-quality.md`; the script itself belongs in `research/exploratory_notebooks/` if revived, not in production code. |
| `FINAL_SUMMARY.md`, `TECHNICAL_SUMMARY.md`, `TASK_COMPLETION_SUMMARY.md`, `PHASE_1_SETUP.md`, `PHASE_1_COMPLETE.md`, `QUICK_REFERENCE.md`, `AGENTS.md` | Root-level phase-summary sprawl — the exact "looks like an agent work log, not a maintained product" issue flagged in the repositioning review. Substantive content merged into `docs/`. |
| `GOVERNANCE_AUDIT.md`, `DESIGN_DECISIONS.md` | Superseded by `docs/limitations.md` and `docs/decisions/001`–`006` — same substance, reorganized into the target `docs/` structure instead of two flat root files. |
| `models/*.pkl`, `models/*.npy`, `results/pr_curve_comparison.html` | Stale, and **incompatible** with the new feature set — the old artifacts were trained on 4 PascalCase-named features (`DegradationRate`, `StintAgeSquared`, `RaceProgress`, `PaceDelta`); the new registry uses 6 snake_case features. Loading them into the new app would silently mismatch, not just look outdated. Regenerate with `make ingest && make train` — see `.gitignore` for why they're not committed going forward (the new repo's positioning is "dynamic model training," not a checked-in binary). |
| `lightgbm`, `matplotlib`, `pymysql`, `pyodbc`, `pydantic-settings`, `python-json-logger`, `black`, `mypy`, `pre-commit` (from `requirements.txt`) | Unused by anything in `src/f1_pit_window` (LightGBM and matplotlib were never actually imported by shipped code; the multi-DB drivers went with `config.py`; black/mypy/pre-commit were never wired into CI). Trimmed rather than carried forward as unused weight. |
| `.github/workflows/main_fastf1-pit-strategy-optimization.yml` (Azure deploy) | Tied to Azure App Service secrets scoped to the old repo/resource name. Not reusable as-is; add a deployment workflow for wherever the new repo actually gets hosted, rather than porting broken Azure secrets references. |

## What still needs real work before this is "done"

Honest gaps, not silently absent:

1. **No data has been ingested into this branch.** `models/`, `results/` are
   empty (`.gitkeep` only). Run `make db-up && make ingest && make train`
   against a real `DATABASE_URL` to produce actual artifacts and actual
   reported metrics — `docs/model-evaluation.md` describes the evaluation
   methodology, not numbers, because there are no numbers yet from this
   pipeline.
2. **`opponent_pit_window_signal` is unimplemented** — a real modeling
   decision deferred, not a bug (`docs/decisions/005-strategy-context-features.md`).
3. **`db/metrics_view.sql` is generated but not yet applied to a live DB**
   automatically — `make apply-view` after the first ingest; see the note
   in `docker-compose.yml` for why it isn't wired into container boot.
4. **No group-wise calibration, no circuit-holdout evaluation** — listed as
   gaps in `docs/model-evaluation.md` and `docs/decisions/002-temporal-validation.md`.

## How this was moved

The rewrite was staged on the prior repo's `new-repo-scaffold` branch and
applied here as a single consolidating commit, so this repo's own initial
commit stays reachable rather than being force-replaced. Carried over in that
step, beyond the table above: ADR `007-monolith-architecture.md`, the
benchmarking subsystem, the scheduled-retrain workflow, and `AGENTS.md`.

Deliberately **not** carried over from the prior repo's `main`: its
`models/*.pkl`, `*.npy`, and reported metrics. Those were trained on
synthetic data (`load_real_data.py`'s `create_realistic_race()`, built on
`np.random`) against four PascalCase features, so they are both unreproducible
and incompatible with the current nine-metric registry. Re-run `make ingest &&
make train` against real FastF1 data to produce numbers this project can
actually stand behind.
