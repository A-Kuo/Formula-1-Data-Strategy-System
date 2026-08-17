# 006 — The prior repo had two parallel, inconsistent pipelines

## Context

A repositioning review of the prior repo flagged inconsistent headline
metrics reported in different places (0.432 F1 / 0.760 ROC-AUC vs. 0.449 F1
/ 0.841 ROC-AUC). Tracing *why* surfaced a structural issue, not a
measurement error.

## What was found

Two entirely separate pipelines existed, never reconciled:

- `pipeline.py` → 4 synthetic features (`DegradationRate`,
  `StintAgeSquared`, `RaceProgress`, `PaceDelta`) → `models/xgboost_model.pkl`
  → `streamlit_app_enhanced.py`.
- `load_real_data.py` / `feature_engineering_real.py` /
  `model_comparison_enhanced.py` → 14 features (including the fabricated
  `GapToLeader`/`GapToCarInFront`/`PitDeltaEstimated` constants documented
  in `004-canonical-metric-schema.md`) → `models/random_forest.pkl`,
  `models/xgboost.pkl`, `models/logistic_regression.pkl`,
  `results/model_comparison.csv` → `streamlit_app.py`.

**Only the first pipeline's artifacts existed in the committed repo.**
`streamlit_app.py` would raise on load — its required model files were
never regenerated after the pipeline was rebuilt around the 4-feature
model. It wasn't a second live dashboard giving a second, disagreeing
number; it was dead code that nobody had run end-to-end since the rebuild.

## Decision

This project ships exactly **one** training path
(`scripts/train_model.py`) and **one** dashboard
(`src/f1_pit_window/app/streamlit_app.py`). The 14-feature path,
`streamlit_app.py`, and the duplicate `sql_utils.py` ORM schema (a *third*,
independently-defined database schema — see
`src/f1_pit_window/data/schema.py`'s docstring) are not ported. There is
nothing left in this repository capable of producing a second, disagreeing
headline number, because there is exactly one pipeline that produces one.

## Consequences

- `docs/model-evaluation.md`'s reported metrics have exactly one source:
  `scripts/train_model.py`'s output, saved to `models/metrics.pkl` and read
  directly by the Streamlit app's Model Review view — no manual
  transcription step where a stale number could survive a retrain.
- Retraining and reporting new numbers is one command
  (`make train`), not a question of which of several scripts is
  "the real one" this week.
