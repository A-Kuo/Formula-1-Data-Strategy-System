"""Calibration tests — Brier score and reliability curve."""

import numpy as np
import pytest

from f1_pit_window.modeling.calibration import brier_score, calibration_by_group, reliability_curve


class TestBrierScore:
    def test_perfect_predictions_score_zero(self):
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.0, 1.0, 0.0, 1.0])
        assert brier_score(y_true, y_proba) == pytest.approx(0.0)

    def test_worst_case_predictions_score_one(self):
        y_true = np.array([0, 1])
        y_proba = np.array([1.0, 0.0])
        assert brier_score(y_true, y_proba) == pytest.approx(1.0)

    def test_constant_prediction_of_true_base_rate_beats_naive_half(self):
        rng = np.random.RandomState(0)
        y_true = (rng.uniform(0, 1, 1000) < 0.15).astype(int)
        base_rate_score = brier_score(y_true, np.full_like(y_true, 0.15, dtype=float))
        naive_half_score = brier_score(y_true, np.full_like(y_true, 0.5, dtype=float))
        assert base_rate_score < naive_half_score


class TestReliabilityCurve:
    def test_raises_on_unknown_strategy(self):
        with pytest.raises(ValueError):
            reliability_curve(np.array([0, 1]), np.array([0.1, 0.9]), strategy="bogus")

    def test_perfectly_calibrated_model_lies_near_the_diagonal(self):
        rng = np.random.RandomState(0)
        y_proba = rng.uniform(0, 1, 5000)
        y_true = (rng.uniform(0, 1, 5000) < y_proba).astype(int)
        curve = reliability_curve(y_true, y_proba, n_bins=5)
        assert np.allclose(curve["mean_predicted"], curve["observed_rate"], atol=0.1)

    def test_bin_counts_sum_to_input_length(self):
        y_true = np.array([0, 1] * 50)
        y_proba = np.linspace(0, 1, 100)
        curve = reliability_curve(y_true, y_proba, n_bins=5)
        assert curve["n"].sum() == 100


class TestCalibrationByGroup:
    def test_small_group_reports_nan(self):
        import pandas as pd
        result = calibration_by_group(np.array([0, 1]), np.array([0.1, 0.9]), pd.Series(["a", "a"]), min_group_size=30)
        assert result["brier_score"].isna().all()
