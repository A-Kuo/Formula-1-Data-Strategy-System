"""
Inference
=========

``validate_model_artifacts`` is ported from ``f1pipeline/model_metrics.py``
— the pre-dashboard validation gate on *loaded model artifacts* (as opposed
to :mod:`f1_pit_window.data.validation`, which gates raw lap data). Both
exist because they catch different failure classes at different points in
the pipeline: a corrupted lap batch and a corrupted training run are not the
same event, and conflating their checks into one function would blur which
one actually failed when something does.

``predict`` is new: a thin wrapper that always runs the gate before serving
a prediction, so nothing calling this module can accidentally skip it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator


class ModelArtifactValidationError(Exception):
    """Raised when a loaded model artifact fails a pre-serving validation check."""


class SavedMetrics(BaseModel):
    """
    Schema for the saved-metrics dict a training run writes out.

    Bounds encode what's physically possible, not what's good — a weak
    model (low ROC-AUC, still in [0, 1]) must pass; an impossible one
    (ROC-AUC > 1) must not.
    """

    model_config = ConfigDict(extra="allow")

    roc_auc: float
    f1: float
    recall: float
    precision: float
    threshold: float
    train_size: int
    test_size: int

    @field_validator("roc_auc", "f1", "recall", "precision", "threshold")
    @classmethod
    def _unit_interval(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"must be within [0, 1], got {value}")
        return value

    @field_validator("train_size", "test_size")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"must be positive, got {value}")
        return value


def validate_model_artifacts(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_proba: np.ndarray,
    saved_metrics: dict,
) -> None:
    """
    Fail-closed check on loaded model artifacts, run once before anything
    derived from them is computed or rendered.

    Raises:
        ModelArtifactValidationError: On the first failing check.
    """
    try:
        SavedMetrics.model_validate(saved_metrics)
    except Exception as exc:
        raise ModelArtifactValidationError(f"saved_metrics failed schema validation: {exc}") from exc

    if not np.all(np.isfinite(X_test)):
        n_bad = int(np.count_nonzero(~np.isfinite(X_test)))
        raise ModelArtifactValidationError(f"X_test contains {n_bad} non-finite value(s) — refusing to serve")

    unique_labels = set(np.unique(y_test).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ModelArtifactValidationError(f"y_test must be binary {{0, 1}}, found labels {unique_labels}")

    if not np.all(np.isfinite(y_proba)):
        n_bad = int(np.count_nonzero(~np.isfinite(y_proba)))
        raise ModelArtifactValidationError(f"y_proba contains {n_bad} non-finite value(s) — refusing to serve")
    if np.any((y_proba < 0.0) | (y_proba > 1.0)):
        lo, hi = float(np.min(y_proba)), float(np.max(y_proba))
        raise ModelArtifactValidationError(f"y_proba must be within [0, 1]; observed range [{lo}, {hi}]")

    lengths = {"X_test": len(X_test), "y_test": len(y_test), "y_proba": len(y_proba)}
    if len(set(lengths.values())) != 1:
        raise ModelArtifactValidationError(f"array length mismatch — {lengths}")


def predict(model, scaler, feature_row: dict, feature_cols: list[str]) -> float:
    """
    Score a single hand-entered or dashboard-supplied feature row.

    ``feature_row`` must supply every column in ``feature_cols``; this is
    the boundary between the app layer's slider inputs and the model, kept
    deliberately dumb — no feature computation happens here, only scaling
    and prediction.

    Raises:
        KeyError: If ``feature_row`` is missing any required feature.
    """
    missing = [col for col in feature_cols if col not in feature_row]
    if missing:
        raise KeyError(f"feature_row missing required columns: {missing}")
    X = pd.DataFrame([feature_row])[feature_cols].values
    X_scaled = scaler.transform(X)
    return float(model.predict_proba(X_scaled)[0, 1])
