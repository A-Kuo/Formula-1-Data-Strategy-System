"""
Training
========

Trains the pit-window classifier on features produced by
:mod:`f1_pit_window.features.build_features` — never on an inline
reimplementation of the same math. This is the direct fix for the finding in
``docs/`` / ``docs/decisions/``: the prior repo's ``pipeline.py``,
``feature_engineering.py``, ``feature_engineering_real.py``, and
``load_real_data.py`` each computed ``DegradationRate`` independently. There
is exactly one implementation now, and this module imports it rather than
rewriting it.

Target definition unchanged from the prior repo: binary "does this driver
pit within the next 5 laps," which is a **near-term pit-window likelihood
estimate**, not a race-strategy optimizer — see the root ``README.md`` for
why that distinction is stated up front rather than left implicit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from f1_pit_window.features.build_features import compute_all

logger = logging.getLogger(__name__)

SEED = 42

#: Default feature set: the four original governed metrics plus the two
#: strategy-context features that are real (gap_closing_rate,
#: relative_pace_delta). pit_lane_loss_by_circuit is deliberately excluded
#: from the default set — it requires a fitted lookup table (see
#: build_features.build_circuit_pit_loss_table), which callers must build
#: from their own training split and pass in explicitly, not something this
#: module can assume. opponent_pit_window_signal is not implemented at all
#: (see build_features.py) and cannot appear here.
DEFAULT_FEATURE_COLS: tuple[str, ...] = (
    "degradation_rate",
    "stint_age_squared",
    "race_progress",
    "pace_delta",
    "gap_closing_rate",
    "relative_pace_delta",
)

TARGET_COL = "pit_next_5_laps"


def compute_target(frame: pd.DataFrame, lookahead_laps: int = 5) -> pd.Series:
    """
    Binary target: does this driver pit within the next ``lookahead_laps`` laps?

    Computed per (session_key, driver_number) trace, looking strictly
    forward from the current lap — this is the one place in the pipeline
    where getting the direction of a ``shift`` wrong creates a silent
    label-leakage bug (looking backward would let the model "predict" a pit
    stop using laps that happened after it), so it's isolated in its own
    tested function rather than inlined into feature engineering.
    """
    if "is_pit_lap" not in frame.columns:
        raise KeyError("compute_target requires 'is_pit_lap' (see f1_pit_window.data.cleaning.normalize_laps)")
    ordered = frame.sort_values(["session_key", "driver_number", "lap_number"])
    target = ordered.groupby(["session_key", "driver_number"], sort=False)["is_pit_lap"].transform(
        lambda s: s.shift(-1).rolling(lookahead_laps, min_periods=1).max().fillna(0).astype(bool)
    )
    return target.reindex(frame.index).astype(int)


@dataclass(frozen=True)
class TrainingSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    feature_cols: list[str]


def prepare_training_split(
    laps: pd.DataFrame,
    train_years: list[int],
    test_years: list[int],
    feature_cols: list[str] | None = None,
) -> TrainingSplit:
    """
    Compute governed features + target, split by year (temporal holdout —
    train on earlier seasons, test on a later, fully held-out one; never a
    random shuffle, which would leak future-season information into
    training), and fit a StandardScaler on the training fold only.

    ``laps`` must already carry a ``year`` column (joined from race metadata)
    and be normalized (see :mod:`f1_pit_window.data.cleaning`).
    """
    feature_cols = list(feature_cols or DEFAULT_FEATURE_COLS)
    features = compute_all(laps)
    working = pd.concat([laps.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    working[TARGET_COL] = compute_target(working).values

    train_laps = working[working["year"].isin(train_years)]
    test_laps = working[working["year"].isin(test_years)]
    logger.info("Train: %s laps (%s), Test: %s laps (%s)", len(train_laps), train_years, len(test_laps), test_years)

    X_train = train_laps[feature_cols].fillna(0).values
    y_train = train_laps[TARGET_COL].values
    X_test = test_laps[feature_cols].fillna(0).values
    y_test = test_laps[TARGET_COL].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Pit rate train: %.1f%%, test: %.1f%%", y_train.mean() * 100, y_test.mean() * 100)
    return TrainingSplit(X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols)


def candidate_models(y_train: np.ndarray, seed: int = SEED) -> dict:
    """The three-model comparison this project reports: LR baseline, RF, XGBoost."""
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, eval_metric="auc", random_state=seed, verbosity=0,
        ),
    }


def cross_validate(
    models: dict, X_train: np.ndarray, y_train: np.ndarray, n_splits: int = 5, seed: int = SEED,
) -> dict[str, float]:
    """5-fold stratified CV ROC-AUC for each candidate model."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = {}
    for name, model in models.items():
        result = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        scores[name] = float(result.mean())
        logger.info("  %-22s: %.4f ± %.4f", name, result.mean(), result.std())
    return scores


def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1") -> float:
    """Grid-search the decision threshold that maximizes F1 (a caller-supplied metric fn is not yet supported)."""
    from sklearn.metrics import f1_score

    best_score, best_tau = 0.0, 0.5
    for tau in np.arange(0.1, 0.9, 0.01):
        score = f1_score(y_true, (y_proba >= tau).astype(int), zero_division=0)
        if score > best_score:
            best_score, best_tau = score, float(tau)
    return best_tau
