# 003 — The decision threshold is a policy choice, not a model property

## Context

`predict_proba` returns a probability; turning that into a "flag this lap"
decision requires a threshold, and the choice of threshold trades false
positives (analyst attention spent on nothing) against false negatives (a
real pit window missed). There is no threshold that is objectively correct
— it depends on which error is more expensive in context, a judgment call
this project doesn't get to make on the user's behalf silently.

## Decision

`f1_pit_window.modeling.train.tune_threshold` grid-searches for the
threshold that maximizes F1 on the held-out set — a reasonable *default*,
not a claim of optimality for every use case. The Streamlit app's Decision
Policy view exposes the full precision/recall/F1-vs-threshold curve and
three named decision styles (Coverage-first, Balanced, Intervention-first)
with their implied cost assumptions stated explicitly, rather than shipping
a single hardcoded number with no context.

## Consequences

- An analyst adjusting the threshold slider sees the tradeoff they're
  making, not just a number moving.
- The shipped default threshold is documented as "F1-maximizing on this
  held-out set," not as "the correct threshold" — a claim this project has
  no basis to make, since the correct threshold depends on a cost function
  nobody has specified.
- If a real deployment ever attaches an actual cost model (e.g., "a false
  alarm costs an analyst N minutes, a missed window costs M"), the
  threshold should be chosen by that cost model directly, using
  `f1_pit_window.modeling.evaluate.threshold_sweep`'s full curve, not
  re-derived from F1 alone.
