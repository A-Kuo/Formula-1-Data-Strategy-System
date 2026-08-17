"""
Drift monitoring
================

New in this project's re-scoping — the prior repo had no monitoring at all.
Implements Population Stability Index (PSI), a standard, well-understood
technique for comparing a reference distribution (training data) against a
current one (a new batch), per feature. No custom statistics invented here.

Deliberately scoped to **detection**, not automated response: this module
answers "has feature X's distribution shifted since training," not "retrain
the model" or "reject this batch." Wiring PSI output into an automatic
retraining trigger is a real design decision (how often, on what threshold,
with what human-in-the-loop checkpoint) that belongs in an operational
runbook, not hardcoded into a statistics function — see
``docs/decisions/`` if that wiring is added later.

Conventional PSI severity bands (used throughout, not just invented here):

* PSI < 0.10 — no significant shift
* 0.10 ≤ PSI < 0.25 — moderate shift, worth investigating
* PSI ≥ 0.25 — significant shift, treat predictions on this feature with suspicion
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_STABLE = 0.10
PSI_MODERATE = 0.25


def population_stability_index(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    PSI between a reference and current sample of the same feature.

    Bin edges are quantiles of the *reference* distribution (not the
    current one, and not the pooled data) — PSI measures how much the
    current sample has moved relative to the distribution the model was
    trained on, so the bins must be fixed by that reference.

    A bin with zero mass in either sample is given a small floor (rather
    than 0, which would make PSI undefined via a division by zero, or an
    unbounded contribution from a near-zero denominator) — this is the
    standard treatment, not a shortcut specific to this implementation.

    Raises:
        ValueError: If either input is empty.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("population_stability_index requires non-empty, finite input on both sides")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        # Reference feature has (near-)zero variance — no meaningful bins to compare.
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    floor = 1e-4
    ref_pct = np.maximum(ref_counts / len(reference), floor)
    cur_pct = np.maximum(cur_counts / len(current), floor)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def severity(psi: float) -> str:
    """Human-readable band for a PSI value — see module docstring for the thresholds."""
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_MODERATE:
        return "moderate"
    return "significant"


def feature_drift_report(
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    feature_cols: list[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    PSI per feature, comparing ``current_frame`` against ``reference_frame``
    (typically: the training set vs. a new batch of live data).

    Columns present in ``feature_cols`` but missing from either frame are
    skipped with a logged reason rather than raising — a monitoring report
    with 11 of 12 features is still useful; refusing to run it over one
    missing column is not the fail-closed behavior that matters here (that's
    :mod:`f1_pit_window.data.validation`'s job, upstream of this).
    """
    rows = []
    for column in feature_cols:
        if column not in reference_frame.columns or column not in current_frame.columns:
            rows.append({"feature": column, "psi": np.nan, "severity": "unavailable (column missing)"})
            continue
        psi = population_stability_index(
            reference_frame[column].to_numpy(dtype=float),
            current_frame[column].to_numpy(dtype=float),
            n_bins=n_bins,
        )
        rows.append({"feature": column, "psi": psi, "severity": severity(psi)})

    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)
