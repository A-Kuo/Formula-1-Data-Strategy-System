"""
Calibration
===========

New in this project's re-scoping. The prior repo's dashboard displayed a
"pit probability" without ever checking whether that probability was
calibrated — a 0.76 ROC-AUC model can rank-order predictions well while its
probability *values* are still systematically over- or under-confident,
which matters directly for a threshold-based decision UI: if "70% probable"
doesn't mean "pits about 70% of the time when this is predicted," the
threshold slider is calibrated against a number that doesn't mean what it
claims to.

Uses :func:`sklearn.calibration.calibration_curve` and
:func:`sklearn.metrics.brier_score_loss` — no custom statistics here, this
is standard, well-understood machinery, deliberately not reinvented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Mean squared error between predicted probability and the binary outcome.

    0.0 is perfect; 0.25 is what a constant 0.5 prediction scores on a
    balanced target — a useful mental anchor when this project's target is
    imbalanced (~15% positive rate) and 0.25 is not the right reference
    point here (a model predicting the base rate everywhere scores much
    lower than 0.25 on this target; compare against that baseline, not 0.25).
    """
    return float(brier_score_loss(y_true, y_proba))


def reliability_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """
    Predicted-probability bins vs. observed positive rate in each bin.

    ``strategy="quantile"`` (the default) bins by count rather than by
    probability range — with a ~15% positive rate and heavy mass near low
    probabilities, uniform-width bins put almost all the data in the first
    bin and leave the high-probability bins nearly empty; quantile binning
    keeps every bin populated enough to be a meaningful comparison.

    Returns:
        DataFrame with ``mean_predicted``, ``observed_rate``, and ``n`` per
        bin, one row per bin that received at least one prediction (an empty
        bin contributes no row rather than a fabricated NaN row).

    Raises:
        ValueError: If ``strategy`` is not ``"quantile"`` or ``"uniform"``.
    """
    if strategy not in ("quantile", "uniform"):
        raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if strategy == "quantile":
        bins = pd.qcut(y_proba, q=n_bins, duplicates="drop")
    else:
        bins = pd.cut(y_proba, bins=n_bins)

    frame = pd.DataFrame({"y_true": y_true, "y_proba": y_proba, "bin": bins})
    grouped = frame.groupby("bin", observed=True)
    return pd.DataFrame({
        "mean_predicted": grouped["y_proba"].mean(),
        "observed_rate": grouped["y_true"].mean(),
        "n": grouped.size(),
    }).reset_index(drop=True)


def calibration_by_group(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    groups: pd.Series,
    min_group_size: int = 30,
) -> pd.DataFrame:
    """
    Brier score sliced by an arbitrary grouping column (tyre compound,
    race-progress bucket) — a model can be well-calibrated in aggregate
    while badly miscalibrated for, say, wet-adjacent laps that are rare in
    training. Groups under ``min_group_size`` report ``NaN`` rather than a
    Brier score computed on too few points to trust.
    """
    groups = pd.Series(groups).reset_index(drop=True)
    frame = pd.DataFrame({"y_true": np.asarray(y_true), "y_proba": np.asarray(y_proba), "group": groups.values})
    rows = []
    for group_value, sub in frame.groupby("group", dropna=False):
        n = len(sub)
        score = brier_score(sub["y_true"], sub["y_proba"]) if n >= min_group_size else float("nan")
        rows.append({"group": group_value, "n": n, "brier_score": score})
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
