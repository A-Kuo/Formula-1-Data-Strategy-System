# Methodology

## Problem framing

This project estimates whether a driver is likely to pit **within the next 5
laps**, given tire-degradation and race-state features at the current lap. It
is a **binary classification / decision-support** problem, not a race-strategy
optimizer — it does not jointly model tire choice, fuel load, traffic, or
competitor response over the remainder of a race. See `docs/limitations.md`
for the full scope statement.

## Temporal validation, not random split

Training uses seasons **2018–2023**; the **2024** season is held out
completely and evaluated only once, after model selection is final. This is
deliberate: F1 cars, regulations, and driver lineups change materially
year over year, and a random shuffle-split would let a later-season data
point leak into training, overstating generalization. A temporal holdout
answers the actual deployment question — "does this model work on a season
it has never seen" — a random split does not.

See `docs/decisions/002-temporal-validation.md` for the reasoning in more
detail, including what circuits are and aren't shared between the two splits.

## Target definition

`pit_next_5_laps`: for each (session, driver, lap), `True` if that driver
pits (`is_pit_lap`) on any of the next 5 laps, looking strictly forward.
Computed in `f1_pit_window.modeling.train.compute_target`. See
`docs/decisions/001-target-definition.md` for why "strictly forward" is
the one detail in this function's implementation that would silently leak
labels if gotten wrong.

## Data cleaning

Laps affected by conditions this feature set doesn't observe are excluded
before training — see `docs/data-quality.md` for exact exclusion criteria and
retention rates. The model's behavior under safety-car, VSC, standing-start,
or wet-weather conditions is **unvalidated**, not just untested; this is
stated as a limitation, not hidden.

## Features

Six governed features, each with a versioned definition in
`f1_pit_window.features.build_features.CANONICAL_REGISTRY` — see
`docs/feature-engineering.md`. Four are the project's original tire-
degradation and race-state features (`degradation_rate`, `stint_age_squared`,
`race_progress`, `pace_delta`); two (`gap_closing_rate`,
`relative_pace_delta`) are strategy-context features added to address a
specific error-analysis finding — see `docs/model-evaluation.md`.

## Models compared

Logistic Regression (linear baseline), Random Forest, and XGBoost — 5-fold
stratified cross-validation on the training set, ROC-AUC as the CV selection
metric (appropriate for imbalanced classification; F1/precision/recall are
then reported on the held-out set at a tuned threshold, not used for model
*selection*, since they depend on the threshold itself).

## Threshold policy

The decision threshold is a **policy choice**, not a model output — see
`docs/decisions/003-threshold-policy.md` and the Streamlit app's Decision
Policy view, which exposes the full precision/recall/F1-vs-threshold curve
rather than shipping a single hardcoded cutoff without context.

## Reproducibility

- `scripts/ingest_data.py` — fetch (FastF1) → normalize
  (`f1_pit_window.data.cleaning`) → write.
- `scripts/train_model.py` — load → compute governed features
  (`f1_pit_window.features.build_features`) → temporal split → train → tune
  threshold → save artifacts.
- `pytest tests/` — 120+ tests, 92%+ coverage on business logic, including a
  SQL/pandas parity suite that runs the *generated* SQL view against real
  data, not a separately-maintained approximation of it.
