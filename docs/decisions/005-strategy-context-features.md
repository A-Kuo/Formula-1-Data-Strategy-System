# 005 — Strategy-context features: three built for real, one deferred, one bug caught live

## Context

The prior repo's error analysis found the actual gap: the model sees tire
wear but no tactical context, so it predicts "old tires ⇒ pit" even when a
driver stays out for a strategic reason (undercut/overcut) the feature set
doesn't observe. Four features were proposed to address this:
`gap_closing_rate`, `relative_pace_delta`, `pit_lane_loss_by_circuit`,
`opponent_pit_window_signal`.

## Decision

Three implemented for real, using data this project already ingests — no
new data source needed:

- `gap_closing_rate` — trailing-3-lap average change in *instantaneous*
  gap to leader.
- `relative_pace_delta` — lap time minus the field's median at that lap
  (distinct from the existing self-relative `pace_delta`).
- `pit_lane_loss_by_circuit` — a per-circuit expected pit-stop cost,
  implemented as a **fitted lookup table**
  (`build_circuit_pit_loss_table` + `pit_lane_loss_by_circuit`), kept
  deliberately out of the versioned registry because it requires external
  state (the fitted table) the registry's `compute(frame)` contract
  doesn't support — see `docs/feature-engineering.md`.

`opponent_pit_window_signal` is **not implemented**.
`build_features.opponent_pit_window_signal()` raises `NotImplementedError`
with a docstring explaining why: unlike the other three, it requires joint
state across every driver on track at a given lap — a per-row feature can't
be computed without first knowing (or predicting) every nearby driver's pit
status at the same lap, which is either a circular dependency on this
model's own output or a materially different model architecture
(sequence-to-sequence over the grid, not a per-row classifier). That's a
modeling-architecture decision for a future session, not a missing
afternoon of feature engineering — recorded as a raised exception with an
explanation, not a silently-omitted feature or a fake stub.

## A bug this decision's own implementation caught, live

`gap_closing_rate`'s first draft computed its trailing-3-lap diff on
`gap_to_leader@v2` (the **cumulative** gap). Since the cumulative series is
non-decreasing by construction, its diff measures accumulated gap magnitude
over the window, not closing speed — a driver closing rapidly and a driver
holding a large, steady gap can produce similar values.
`tests/test_build_features.py::test_gap_closing_rate_positive_when_gap_shrinking`
built a case with an unambiguous scenario (driver 2 visibly closing on a
constant-pace leader, lap over lap) and got a *negative* closing rate back
— caught on the first run. Fixed by switching to `gap_to_leader@v1` (the
instantaneous per-lap gap), whose diff actually measures rate of change.
