# AGENTS.md

## Cursor Cloud specific instructions

Project: **F1 Pit-Window Decision Support** — a Python 3.11 ML pipeline
(FastF1 → Postgres → scikit-learn/XGBoost → Streamlit dashboard). See
`README.md` for the full quick-start and `make help` for all targets.

### Environment layout (provided by the base snapshot)

- **Python 3.11** is required (pinned deps predate Python 3.12 wheels). The
  system default `python3` is 3.12 — do **not** use it. Use the project
  virtualenv at `.venv/` (built with `python3.11`). The startup update script
  refreshes it from `requirements.txt`.
- Run project commands through the venv, e.g. `.venv/bin/pytest`,
  `.venv/bin/flake8`, `.venv/bin/streamlit`, `.venv/bin/python`. The `make`
  targets call bare `pytest`/`flake8`/`streamlit`/`python`, so either activate
  the venv first (`source .venv/bin/activate`) or call the `.venv/bin/*`
  entrypoints directly.
- **PostgreSQL 16** is installed but is **not auto-started** on boot. Start it
  once per session with `sudo pg_ctlcluster 16 main start`. Credentials:
  user `postgres`, password `postgres`; databases `f1_pit_db` and
  `f1_pit_test` exist. Connection string:
  `postgresql://postgres:postgres@localhost:5432/f1_pit_db`.

### Lint / test / build-run

- Lint + SQL-drift check: `.venv/bin/flake8 src tests scripts/generate_metrics_view.py`
  and `.venv/bin/python scripts/generate_metrics_view.py --check` (together:
  `make lint` + `make view-check`).
- Tests: `.venv/bin/pytest tests/` (123 pass, 1 skipped, ~92% coverage,
  coverage-gated at 85%). **No database is required** — the SQL-parity tests
  run against in-memory SQLite. Coverage gate applies to the whole suite, so a
  single-file run will "fail" the 85% gate; run the full suite to gate.
- The extra live-Postgres parity test only runs when `DATABASE_URL` starts
  with `postgresql://`; set it (after starting Postgres) to exercise that path.
- App (dev mode): `.venv/bin/streamlit run src/f1_pit_window/app/streamlit_app.py`
  (serves on `:8501`, `/_stcore/health` returns 200).

### Running the dashboard end to end (non-obvious)

The Streamlit app refuses to render until trained model artifacts exist in
`models/` (`xgboost_model.pkl`, `scaler.pkl`, `metrics.pkl`, `X_test_scaled.npy`,
`y_test.npy`). These are **build outputs, gitignored, and absent on a fresh
checkout** (this branch is a scaffold — see `README.md` "Status"). Full real
flow (Postgres must be running; first ingest downloads from the FastF1 network
and caches into `.cache/`, so it is slow the first time and fast afterwards):

```
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/f1_pit_db \
  .venv/bin/python scripts/ingest_data.py --years 2022,2023,2024 --races 1,2,3,4,5,6,7,8
DATABASE_URL=... .venv/bin/python scripts/apply_metrics_view.py
DATABASE_URL=... .venv/bin/python scripts/train_model.py --train-years 2022,2023 --test-years 2024
```

`apply_metrics_view.py` creates the `canonical_lap_metrics` view, which depends
on `laps`; to re-ingest from scratch, drop it first
(`DROP VIEW IF EXISTS canonical_lap_metrics CASCADE;` then the tables), since a
plain `DROP TABLE laps` fails while the view exists.

### FastF1 ingestion fixes (already applied on this branch)

Two pre-existing bugs blocked real ingestion with `fastf1==3.8.3` and are fixed
in `ingestion/fastf1_client.py` and `data/schema.py`: `fetch_session` read
`session.event['Season']`/`['Circuit']` (keys that no longer exist — use
`session.event.year` and `session.event['Location']`), and `laps.tyre_compound`
was `String(10)`, too small for the contract's own `INTERMEDIATE` value (wet
races), now `String(20)`. Ingestion of 2022–2024 real data works end to end.

### Dependency note

`requirements.txt` was corrected during setup: the previous `numpy==1.24.3` /
`pandas==2.0.3` pins were below `fastf1==3.8.3`'s floors (`numpy>=1.26.0`,
`pandas>=2.1.1`), making the requirement set impossible to install. They are
now `numpy==1.26.4` / `pandas==2.1.4` (still under streamlit's `numpy<2`).
