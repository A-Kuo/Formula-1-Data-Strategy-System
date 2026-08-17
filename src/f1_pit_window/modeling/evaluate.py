"""
Evaluation
==========

``threshold_sweep`` is ported from ``f1pipeline/model_metrics.py`` (formerly
duplicated inline in both of the prior repo's Streamlit files — see
``docs/decisions/``). ``group_wise_metrics`` is new: the prior repo reported
a single aggregate ROC-AUC/F1 across the entire held-out season and nothing
else, which hides exactly the kind of failure a reviewer will ask about —
does this model work as well at a street circuit as a permanent one, does it
degrade on tyre compounds under-represented in training. This module reports
metrics sliced by a grouping column instead of only in aggregate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def threshold_sweep(y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Precision/recall/F1 at each candidate decision threshold."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({"Threshold": float(threshold), "Precision": precision, "Recall": recall, "F1": f1})
    return pd.DataFrame(rows)


def group_wise_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    groups: pd.Series,
    threshold: float = 0.6,
    min_group_size: int = 30,
) -> pd.DataFrame:
    """
    ROC-AUC / precision / recall / F1 sliced by an arbitrary grouping column
    (circuit, season, tyre compound, race-progress bucket — whatever the
    caller passes as ``groups``, aligned by position to ``y_true``/``y_proba``).

    Groups smaller than ``min_group_size`` are reported with metrics as
    ``NaN`` rather than a misleadingly precise number computed on a handful
    of laps — an ROC-AUC from 8 examples is noise, not signal, and showing
    it as a real number invites over-interpreting it.

    Raises:
        ValueError: If ``y_true``, ``y_proba``, and ``groups`` have mismatched lengths.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    groups = pd.Series(groups).reset_index(drop=True)
    if not (len(y_true) == len(y_proba) == len(groups)):
        raise ValueError(f"length mismatch: y_true={len(y_true)}, y_proba={len(y_proba)}, groups={len(groups)}")

    frame = pd.DataFrame({"y_true": y_true, "y_proba": y_proba, "group": groups.values})
    rows = []
    for group_value, sub in frame.groupby("group", dropna=False):
        n = len(sub)
        y_pred = (sub["y_proba"] >= threshold).astype(int)
        row = {"group": group_value, "n": n}
        if n < min_group_size or sub["y_true"].nunique() < 2:
            row.update({"roc_auc": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan})
        else:
            row["roc_auc"] = roc_auc_score(sub["y_true"], sub["y_proba"])
            row["precision"] = precision_score(sub["y_true"], y_pred, zero_division=0)
            row["recall"] = recall_score(sub["y_true"], y_pred, zero_division=0)
            row["f1"] = f1_score(sub["y_true"], y_pred, zero_division=0)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
