# 002 — Temporal holdout, not a random split

## Context

F1 seasons change materially year over year: regulations, car performance
envelopes, driver lineups, even circuit layouts. A random shuffle-split
across all seasons would let information from a later season leak into
training for an earlier one — the model could learn something true only of
2024 and get "credit" for it while being evaluated on 2024 laps that were
in its training set.

## Decision

Train on 2018–2023, evaluate once on the fully held-out 2024 season
(`f1_pit_window.modeling.train.prepare_training_split`,
`train_years`/`test_years` parameters). No shuffling across the boundary.

## Consequences, stated plainly

- This measures generalization **across seasons**, which is the real
  deployment question — does this model still work on a season it has
  never seen a single lap of.
- It does **not** measure generalization across **circuits**: most 2024
  circuits also appear in 2018–2023 training data, so a circuit the model
  has genuinely never encountered in any form is not covered by this split.
  `docs/limitations.md` states this explicitly rather than letting "temporal
  holdout" imply more rigor than it delivers.
- A circuit-holdout variant (train on N-1 circuits, test on the Nth,
  repeated) would answer the circuit-generalization question directly and
  is a reasonable next evaluation to add — not implemented here, listed as
  a gap rather than silently absent.
