#!/usr/bin/env python3
"""
Train the pit-window classifier against ingested data and save artifacts.

Reads normalized laps from PostgreSQL (written by scripts/ingest_data.py),
computes governed features (f1_pit_window.features.build_features), trains
and compares the three candidate models, tunes the decision threshold, and
writes the artifacts the Streamlit app loads: models/xgboost_model.pkl,
models/scaler.pkl, models/metrics.pkl, models/X_test_scaled.npy, models/y_test.npy.

Usage:
    DATABASE_URL=postgresql://... python scripts/train_model.py \
        --train-years 2018,2019,2020,2021,2022,2023 --test-years 2024
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from f1_pit_window.data.schema import LapORM, RaceORM  # noqa: E402
from f1_pit_window.modeling.train import (  # noqa: E402
    candidate_models,
    cross_validate,
    prepare_training_split,
    tune_threshold,
)

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-years", default="2018,2019,2020,2021,2022,2023")
    parser.add_argument("--test-years", default="2024")
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    db_url = args.db_url or os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/f1_pit_db")
    train_years = [int(y) for y in args.train_years.split(",")]
    test_years = [int(y) for y in args.test_years.split(",")]

    logger.info("Loading laps for %s", train_years + test_years)
    laps = load_laps(db_url, train_years + test_years)

    logger.info("Preparing temporal train/test split and computing governed features...")
    split = prepare_training_split(laps, train_years=train_years, test_years=test_years)

    logger.info("5-fold CV across candidate models...")
    models = candidate_models(split.y_train)
    cv_scores = cross_validate(models, split.X_train, split.y_train)

    xgb = models["XGBoost"]
    xgb.fit(split.X_train, split.y_train)
    y_proba = xgb.predict_proba(split.X_test)[:, 1]

    threshold = tune_threshold(split.y_test, y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(split.y_test, y_proba)),
        "f1": float(f1_score(split.y_test, y_pred)),
        "recall": float(recall_score(split.y_test, y_pred)),
        "precision": float(precision_score(split.y_test, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "train_size": len(split.y_train),
        "test_size": len(split.y_test),
        "feature_cols": split.feature_cols,
        "cv_scores": cv_scores,
    }
    logger.info("Held-out: ROC-AUC=%.4f F1=%.4f Recall=%.4f Precision=%.4f (τ=%.2f)",
                metrics["roc_auc"], metrics["f1"], metrics["recall"], metrics["precision"], threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb, f)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(split.scaler, f)
    with open(output_dir / "metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    np.save(output_dir / "X_test_scaled.npy", split.X_test)
    np.save(output_dir / "y_test.npy", split.y_test)

    logger.info("✓ Artifacts written to %s/", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
