#!/usr/bin/env python3
"""
One-off race-level holdout proof run — NOT the documented cross-season protocol.

docs/decisions/002-temporal-validation.md specifies training on 2018-2023 and
testing on held-out 2024. This environment only has FastF1 cache data for
three 2024 races (network access to fetch earlier seasons is blocked by this
sandbox's egress policy), so that protocol cannot be executed here. This
script is a smaller, honestly-labeled substitute: it holds out one full race
(chronologically the latest of the three) rather than an earlier lap-level
or year-level split, using the same governed features, the same three-model
comparison, and the same threshold-tuning as scripts/train_model.py — just
with the split criterion swapped from "year" to "race_id" for this run only.

Train: Monaco GP + Italian GP (2024, race_id 1 and 2)
Test:  Singapore GP (2024, race_id 3) — chronologically the latest of the
       three, so this is still a forward-in-time holdout, just within one
       season instead of across seasons.

Writes the same artifact contract as scripts/train_model.py
(models/xgboost_model.pkl, scaler.pkl, metrics.pkl, X_test_scaled.npy,
y_test.npy) so scripts/validate_artifacts.py and the Streamlit app can use
the output, but metrics.pkl carries an explicit "protocol" field so nothing
downstream can mistake this for the cross-season result.

Usage:
    DATABASE_URL=postgresql://... python scripts/train_model_race_holdout.py
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
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from f1_pit_window.data.schema import LapORM, RaceORM  # noqa: E402
from f1_pit_window.modeling.train import (  # noqa: E402
    DEFAULT_FEATURE_COLS,
    TARGET_COL,
    candidate_models,
    compute_target,
    cross_validate,
    tune_threshold,
)
from f1_pit_window.features.build_features import compute_all  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s")
logger = logging.getLogger(__name__)


def load_laps(db_url: str) -> pd.DataFrame:
    engine = create_engine(db_url)
    with Session(engine) as session:
        laps = pd.read_sql(session.query(LapORM).statement, engine)
        races = pd.read_sql(session.query(RaceORM).statement, engine)
    if laps.empty:
        raise ValueError("No data found — run scripts/ingest_data.py first")
    return laps.merge(races[["race_id", "year", "circuit_name", "race_name"]], on="race_id", how="left")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-race-ids", default="1,2")
    parser.add_argument("--test-race-ids", default="3")
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    db_url = args.db_url or os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/f1_pit_db")
    train_race_ids = [int(r) for r in args.train_race_ids.split(",")]
    test_race_ids = [int(r) for r in args.test_race_ids.split(",")]

    logger.info("Loading laps...")
    laps = load_laps(db_url)
    train_names = sorted(laps[laps["race_id"].isin(train_race_ids)]["race_name"].unique())
    test_names = sorted(laps[laps["race_id"].isin(test_race_ids)]["race_name"].unique())
    logger.info("Train races: %s", train_names)
    logger.info("Test races: %s", test_names)

    feature_cols = list(DEFAULT_FEATURE_COLS)
    features = compute_all(laps)
    working = pd.concat([laps.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    working[TARGET_COL] = compute_target(working).values

    train_laps = working[working["race_id"].isin(train_race_ids)]
    test_laps = working[working["race_id"].isin(test_race_ids)]
    logger.info("Train: %s laps, Test: %s laps", len(train_laps), len(test_laps))

    X_train = train_laps[feature_cols].fillna(0).values
    y_train = train_laps[TARGET_COL].values
    X_test = test_laps[feature_cols].fillna(0).values
    y_test = test_laps[TARGET_COL].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Pit rate train: %.1f%%, test: %.1f%%", y_train.mean() * 100, y_test.mean() * 100)

    logger.info("5-fold CV across candidate models...")
    models = candidate_models(y_train)
    cv_scores = cross_validate(models, X_train_scaled, y_train)

    xgb = models["XGBoost"]
    xgb.fit(X_train_scaled, y_train)
    y_proba = xgb.predict_proba(X_test_scaled)[:, 1]

    threshold = tune_threshold(y_test, y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "protocol": "race-level holdout (proof run, NOT the cross-season protocol in docs/decisions/002)",
        "train_races": train_names,
        "test_races": test_names,
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "feature_cols": feature_cols,
        "cv_scores": cv_scores,
    }
    logger.info("Held-out: ROC-AUC=%.4f F1=%.4f Recall=%.4f Precision=%.4f (τ=%.2f)",
                metrics["roc_auc"], metrics["f1"], metrics["recall"], metrics["precision"], threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb, f)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(output_dir / "metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    np.save(output_dir / "X_test_scaled.npy", X_test_scaled)
    np.save(output_dir / "y_test.npy", y_test)
    logger.info("✓ Artifacts written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
