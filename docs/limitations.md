# Limitations

Stated up front, not discovered by a reviewer. This is the model-card
equivalent for this project — read it before quoting a metric from
`docs/model-evaluation.md` without this context attached.

## Scope

**This is a decision-support classifier, not a race-strategy optimizer.**
It estimates whether a driver is likely to pit within the next 5 laps from
tire-degradation and race-state features. It does not jointly optimize tire
compound choice, fuel strategy, traffic, or competitor response — that is a
substantially larger problem (multi-stint joint optimization under
uncertainty), and claiming this system solves it would misrepresent what's
actually implemented.

## What the feature set doesn't observe

- **No opponent pit-window signal.** Whether nearby competitors have pitted
  or are about to is a real, proposed feature
  (`opponent_pit_window_signal`) that is **not implemented** — it raises
  `NotImplementedError` with an explanation, not a silent placeholder. See
  `docs/feature-engineering.md`.
- **No fuel load, no DRS state, no explicit undercut/overcut detection.**
  Many of the false positives this model produces reflect a driver staying
  out on worn tires for a tactical reason (protecting track position,
  setting up an overcut) that isn't in the feature set at all — the model
  isn't wrong about the tire physics, it's blind to the strategy layer on
  top of it.

## Data scope

Trained and evaluated on **clean, dry-race laps only** — safety-car, VSC,
standing-start, and wet-weather laps are excluded from both training and
evaluation (`docs/data-quality.md`). This model's behavior under those
conditions is **unvalidated**, not merely untested: there's no evidence it
degrades gracefully rather than confidently wrong under conditions its
training distribution never contained.

## Validation split

Training: 2018–2023. Test: 2024, evaluated once. This measures
generalization across **seasons**, not across **circuits** — most circuits
in the 2024 test set also appear in the 2018–2023 training set. Whether this
model generalizes to a genuinely new circuit (one it has never seen in any
form) is a different, harder, and currently unanswered question. See
`docs/decisions/002-temporal-validation.md`.

## Threshold

The shipped decision threshold is a **policy choice**, tied to an implicit
cost assumption (a missed pit-window flag vs. a false alarm), not a property
the model discovered. See `docs/decisions/003-threshold-policy.md` and the
Streamlit app's Decision Policy view, which exposes this as an adjustable
choice rather than a hardcoded number.

## Operational maturity

This is a portfolio-grade, reproducible research/decision-support prototype
— containerized, tested, CI-gated — not a production system with model
registry, secrets management, rollback, access control, or live monitoring.
`f1_pit_window.monitoring.drift` implements Population Stability Index drift
*detection*; nothing currently wires it to a live feed or an automated
retraining trigger, and that wiring is a real operational decision (how
often, on what threshold, with what human-in-the-loop checkpoint) deferred
deliberately rather than hardcoded without one.
