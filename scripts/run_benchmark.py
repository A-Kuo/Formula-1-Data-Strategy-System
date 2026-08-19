#!/usr/bin/env python3
"""
Benchmark the pipeline's runtime and data-processing volume.

Times three reproducible, network-independent pipeline stages against
whatever is already ingested in DATABASE_URL — loading laps, computing the
governed feature set, and training the model comparison's best model — and
appends one row per stage to metrics/benchmark_results.csv (git-tracked:
this file is meant to accumulate a visible history of pipeline performance
across runs and commits, not to be regenerated-and-discarded like
models/*.pkl).

Deliberately excludes scripts/ingest_data.py's FastF1 fetch: that stage's
duration is dominated by external network I/O, not this project's own
processing, and is already self-reported by that script's own log line
("Ingestion complete: N races, N laps, N errors"). This benchmark covers
only the deterministic, reproducible in-process stages. It also fits only
the XGBoost model (not the full 3-model CV comparison scripts/train_model.py
runs) so it stays fast enough to run often — model-quality metrics
(ROC-AUC, F1, CV scores) are scripts/train_model.py's job, saved to
models/metrics.pkl; this script records only timing and row counts, never
duplicates them.

Usage:
    DATABASE_URL=postgresql://... python scripts/run_benchmark.py \
        --train-years 2018,2019,2020,2021,2022,2023 --test-years 2024
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from f1_pit_window.data.schema import LapORM, RaceORM  # noqa: E402
from f1_pit_window.features.build_features import compute_all  # noqa: E402
from f1_pit_window.modeling.train import candidate_models, prepare_training_split  # noqa: E402
from f1_pit_window.monitoring.benchmark import BenchmarkRun, write_results  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s")
logger = logging.getLogger(__name__)


def load_laps(db_url: str, years: list[int]) -> pd.DataFrame:
    engine = create_engine(db_url)
    with Session(engine) as session:
        query = session.query(LapORM).join(RaceORM).filter(RaceORM.year.in_(years))
        laps = pd.read_sql(query.statement, engine)
        races = pd.read_sql(session.query(RaceORM).statement, engine)
    if laps.empty:
        raise ValueError(f"No data found for years {years} — run scripts/ingest_data.py first")
    return laps.merge(races[["race_id", "year", "circuit_name"]], on="race_id", how="left")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-years", default="2018,2019,2020,2021,2022,2023")
    parser.add_argument("--test-years", default="2024")
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--output", default="metrics/benchmark_results.csv")
    args = parser.parse_args()

    db_url = args.db_url or os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/f1_pit_db")
    train_years = [int(y) for y in args.train_years.split(",")]
    test_years = [int(y) for y in args.test_years.split(",")]
    years = train_years + test_years

    run = BenchmarkRun()

    logger.info("Benchmarking db_load...")
    start = time.perf_counter()
    laps = load_laps(db_url, years)
    run.record("db_load", len(laps), time.perf_counter() - start, notes=f"years={min(years)}-{max(years)}")

    logger.info("Benchmarking feature_build...")
    start = time.perf_counter()
    features = compute_all(laps)
    run.record("feature_build", len(laps), time.perf_counter() - start, notes=f"features={features.shape[1]}")

    logger.info("Benchmarking train...")
    start = time.perf_counter()
    split = prepare_training_split(laps, train_years=train_years, test_years=test_years)
    models = candidate_models(split.y_train)
    models["XGBoost"].fit(split.X_train, split.y_train)
    duration = time.perf_counter() - start
    run.record(
        "train", len(split.y_train) + len(split.y_test), duration,
        notes=f"train={len(split.y_train)},test={len(split.y_test)}",
    )

    output_path = _REPO_ROOT / args.output
    write_results(run, output_path)
    logger.info("✓ Benchmark recorded to %s (run_id=%s)", output_path, run.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
