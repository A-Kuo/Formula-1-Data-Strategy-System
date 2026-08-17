# 001 — Target definition: `pit_next_5_laps`, strictly forward-looking

## Context

The model predicts whether a driver pits soon. "Soon" needs a precise,
implementable definition, and the direction of the lookahead is the one
detail in this function that silently creates label leakage if gotten
wrong: looking *backward* would let the model "predict" a pit stop using
laps that happened after it.

## Decision

`f1_pit_window.modeling.train.compute_target(frame, lookahead_laps=5)`:
for each (session, driver, lap), `True` if `is_pit_lap` is `True` on any of
the **next** 5 laps for that driver, computed via `shift(-1)` before a
forward-looking rolling window, strictly within each (session, driver)
trace — never crossing into another driver's laps.

## Consequences

- The target is well-defined and testable in isolation
  (`tests/test_train.py::TestComputeTarget`), independent of feature
  computation or model training.
- Explicit tests assert the target does **not** leak across a driver
  boundary and correctly returns 0 for laps with no pit stop anywhere in
  the remainder of the session (not just no pit stop in the next 5 laps of
  a longer trace) — both are the kind of off-by-one/boundary bug that's
  easy to introduce silently in a `groupby().shift()` pipeline.
