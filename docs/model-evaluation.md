# Model Evaluation

## Metrics, and why regression metrics aren't the headline

This is a binary classification problem on an imbalanced target
(~15% positive rate). The primary metrics are **ROC-AUC, PR-AUC, precision,
recall, F1, and a confusion matrix at the tuned threshold** — reported by
`f1_pit_window.modeling.evaluate.threshold_sweep` and computed directly in
`scripts/train_model.py`. MAE/RMSE/R² appeared in the prior repo's headline
table; they can be computed by treating a probability output as a numeric
score, but they aren't the standard basis for evaluating a classifier and
are more likely to confuse a reviewer than clarify anything — they're not
carried forward as primary metrics here.

## The metrics-discrepancy resolution

The prior repo's README reported two different headline numbers in
different places — 0.432 F1 / 0.760 ROC-AUC in one table, 0.449 F1 / 0.841
ROC-AUC in a status line. This was flagged externally as a credibility
problem, and it's a fair catch, but it isn't measurement noise: it's two
**disconnected pipelines** that were never reconciled. One trained a
4-feature XGBoost model (`pipeline.py`); the other, a 14-feature
Random-Forest/XGBoost/Logistic-Regression comparison
(`load_real_data.py` / `feature_engineering_real.py`), whose model artifacts
never actually existed in the committed repo — the dashboard that would have
served them (`streamlit_app.py`) was dead code, silently broken since the
pipeline it depended on was rebuilt out from under it. Full trace in
`docs/decisions/006-dual-pipeline-discovery.md`. This project ships exactly
one training path (`scripts/train_model.py`) and one dashboard
(`src/f1_pit_window/app/streamlit_app.py`) — there is no second number to
disagree with the first.

## Calibration

New in this project. `f1_pit_window.modeling.calibration` computes a Brier
score and a quantile-binned reliability curve — both surfaced in the
Streamlit app's Model Review view. This matters specifically because the
app displays a probability and lets an analyst pick a decision threshold
against it: a model can have a strong ROC-AUC (good rank-ordering) while
its probability *values* are systematically over- or under-confident, which
would make the threshold slider's stated percentages misleading even if the
underlying ranking is sound.

## Group-wise evaluation

New in this project. `f1_pit_window.modeling.evaluate.group_wise_metrics`
slices ROC-AUC/precision/recall/F1 by an arbitrary grouping column —
circuit, season, tyre compound, race-progress bucket. Groups under a
minimum size report `NaN` rather than a misleadingly precise number computed
on a handful of laps. A single aggregate metric across an entire held-out
season, as the prior repo reported, hides exactly the failure a reviewer is
likely to ask about: does this model generalize evenly, or does it work
well on well-represented circuits and poorly on rare ones?

## What's not measured yet

Group-wise calibration by tyre compound, drift monitoring against a live
feed (the machinery exists in `f1_pit_window.monitoring.drift`, but nothing
currently feeds it a live batch to compare against training), and
out-of-sample evaluation on a circuit that appears in *neither* split (the
current split holds out a season, not a circuit — see
`docs/decisions/002-temporal-validation.md` for what that does and doesn't
tell you about generalization to a genuinely new track).
