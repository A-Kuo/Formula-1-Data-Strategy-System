# 007 — Monolith vs. microservices: an architecture options analysis

> **On format**: this follows the general options-analysis-memo structure
> used across technology-strategy consulting engagements — Context →
> Decision Drivers → Options Considered → Trade-off Matrix →
> Recommendation → Risks → Revisit Triggers. It is not a reproduction of
> any specific firm's proprietary template; stating that plainly here
> matches this project's own standard of not overclaiming
> (`docs/decisions/003`'s treatment of the decision threshold takes the
> same approach: state the real basis for a claim, not a more impressive-
> sounding one).

## Executive summary

**Recommendation: a modular monolith** — one Python package, one Docker
image, one running process — over a microservice split. This is the
architecture already built (`src/f1_pit_window/`, one `Dockerfile`, one
`docker-compose.yml`), and this memo is the record of *why*, weighed
explicitly against the alternative rather than assumed. Both options would
technically work: the project could be built either way and would
function correctly. The decision is about fit for this project's actual
context, not a correctness judgment about either approach in general.

## Business context

This is a solo-maintainer portfolio project: one person builds it, runs
it, and (for now) is its only real "user" besides whoever reviews it.
Current requirements are a Streamlit dashboard, a Postgres store, batch
FastF1 data ingestion, and periodic model retraining
(`docs/decisions/002-temporal-validation.md`). There is no multi-team
ownership, no independent scaling requirement, and no external client
depending on any one component in isolation.

The question is worth analyzing explicitly rather than defaulting to
"microservices are best practice" by reflex, because that reflex is
usually imported from a different context: large platforms — food-delivery
apps like Uber Eats or DoorDash are a familiar example — split order
intake, restaurant/driver dispatch matching, live location tracking,
payments, and notifications into independently deployed, independently
scaled services, often coordinated through a message queue. That split
earns its cost there because each of those domains has a genuinely
different scaling curve (driver-location pings are high-frequency,
low-latency; payment processing is low-frequency, high-consistency), is
owned by a separate team shipping on its own release schedule, and needs
to keep functioning in isolation if another part fails (a payments outage
shouldn't stop live tracking). None of those conditions hold here.

## Decision drivers

| Driver | Why it matters for this project |
|---|---|
| Operational complexity | Every additional deployable unit is another thing to build, monitor, and keep in sync — with one maintainer, that cost is paid by the same person who pays every other cost. |
| Team/organizational fit | Microservice boundaries earn their keep when they let separate teams ship independently. There is one team here. |
| Actual scalability requirement | A personal-dashboard workload, not millions of concurrent users — no component here has a scaling curve that diverges from the others enough to justify separating it. |
| Development velocity | A single coupled domain (feature registry → model → dashboard) iterates faster as one deployable than as several kept in sync over a network. |
| Domain coupling | The feature registry, the model, and the dashboard are tightly coupled *by design* — `docs/decisions/004-canonical-metric-schema.md` exists specifically because they must share one source of truth. Splitting tightly coupled domains across a network boundary doesn't decouple them; it just adds serialization and latency between two halves of the same problem. |
| Infrastructure cost | A portfolio project should cost close to nothing to run. Each additional service is an additional thing to host. |
| Build effort | Time spent standing up service discovery, inter-service auth, and independent CI/CD pipelines is time not spent on the actual pipeline (feature engineering, validation, calibration) this project is meant to demonstrate. |
| Failure blast radius | With one process, a bug is debuggable by reading one call stack. With N services, the same bug can hide in the boundary between two of them. |

## Options considered

### Option A — Microservices

Concretely, what the food-delivery-app pattern would look like applied
here: a separate ingestion service (FastF1 fetch + normalize), a separate
feature-computation service, a separate model-serving API (accepting
feature vectors over HTTP/gRPC and returning predictions), and a separate
dashboard frontend calling that API — plus, once split, some coordination
mechanism (a message queue or a scheduler) to keep them working together,
independent datastores or schemas per service, and independent
deploy/monitoring for each.

### Option B — Macroservice / monolith

One deployable unit. Every arrow in this project's architecture diagram
(README.md's "Architecture" section) is an in-process Python function
call: ingestion → cleaning → validation → repository → features →
modeling → app, all inside one process, talking to exactly one external
dependency (Postgres) over a connection string.

### Option C — Modular monolith (what is actually built)

Worth naming as its own point on the spectrum, distinct from either
extreme: one deployable unit (like Option B), but internally organized
into clearly bounded modules with a single-direction dependency flow
(`data/` → `features/` → `modeling/` → `app/`, plus `ingestion/` and
`monitoring/` as siblings) rather than an unstructured "big ball of mud."
This is the actual, current shape of `src/f1_pit_window/` — Option B
described loosely as "a monolith" undersells that the internal boundaries
already exist; they're just not network boundaries.

## Trade-off matrix

Qualitative ratings, not fabricated numeric scores — this project already
prefers a stated qualitative judgment over false precision where the
underlying question doesn't have a numeric answer
(`docs/decisions/003-threshold-policy.md` makes the same call about the
decision threshold).

| Driver | Microservices | Modular monolith (chosen) |
|---|---|---|
| Operational complexity | High — N deployables, N monitoring surfaces | Low — one deployable |
| Team/organizational fit | Only pays off with multiple independent teams | Matches a solo maintainer exactly |
| Scalability headroom | High (each component scales independently) | Sufficient for actual load; not built for independent scaling |
| Development velocity | Slower — cross-service changes need coordinated deploys | Faster — one deploy, one test run |
| Handles tight domain coupling | Poor fit — coupling crosses a network boundary | Good fit — coupling stays in-process |
| Infrastructure cost | Higher — more to host and run | Minimal |
| Build effort | High — service discovery, inter-service auth, N pipelines | Low — one Dockerfile, one CI job |
| Failure blast radius / debuggability | Bugs can hide at service boundaries | One call stack to read |

Microservices win outright on exactly one row (scalability headroom) and
tie-or-lose on every other. That headroom isn't needed here.

## Recommendation & rationale

The modular monolith (Option C) is the right choice for this project, not
because microservices are wrong in general — they're the correct answer
for a food-delivery platform's actual constraints — but because none of
the conditions that make that trade-off worthwhile (independent team
ownership, genuinely divergent scaling curves per component, a need for
partial-failure isolation between unrelated domains) apply to a
single-maintainer portfolio project with one tightly coupled data/model/
dashboard domain. Both options would work; only one fits.

## Risks & mitigations

- **Risk**: harder to scale one component independently later, if this
  project's scope ever genuinely grows past personal-dashboard use.
  **Mitigation**: the modular-monolith boundaries already in place
  (`data/`, `features/`, `modeling/`, `app/`) mean extracting one of them
  into its own service later is a bounded refactor, not a rewrite —
  the internal seams already exist, they'd just need a network interface
  added at one of them.
- **Risk**: a single process is a single point of failure.
  **Mitigation**: the app is stateless (all state lives in Postgres),
  so a crash/restart is cheap and doesn't lose data; there's no
  shared-nothing scaling need at this traffic level to justify the added
  complexity of running multiple replicas behind a load balancer today.

## Revisit triggers

This decision should be reopened, not silently overridden, if any of the
following become true:

- A separate consumer needs to call model inference independently of this
  Streamlit dashboard (at that point, `modeling/inference.py`'s `predict()`
  is the natural extraction point — it's already a plain function with no
  dashboard-specific logic mixed in).
- Genuine multi-team ownership emerges, where independent deploy schedules
  for different components would actually reduce coordination cost instead
  of adding it.
- A specific component's load genuinely diverges from the rest (e.g.
  ingestion needing to run continuously while the dashboard stays
  request-driven) enough that scaling them together becomes wasteful.

## Consequences

- A reviewer can trace any prediction end-to-end by reading one call stack
  in one process, not by reconstructing a request across service
  boundaries.
- There is exactly one thing to deploy (`make docker-up`, or the
  `Dockerfile` directly) and exactly one thing to scale if it ever
  mattered — no per-service deployment matrix to keep in sync.
- `research/AGENT_RESEARCH_2026-06-25.md` and
  `research/AGENT_RESEARCH_2026-07-01.md` carry an explicit header
  pointing back to this document, so they read as what they are —
  speculative brainstorming notes — rather than as an implied roadmap.
