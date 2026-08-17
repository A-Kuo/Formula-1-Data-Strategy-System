"""Threshold sweep and group-wise evaluation tests."""

import numpy as np
import pandas as pd
import pytest

from f1_pit_window.modeling.evaluate import group_wise_metrics, threshold_sweep


class TestThresholdSweep:
    def test_perfect_separation_gives_perfect_scores(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        row = threshold_sweep(y_true, y_proba, np.array([0.5])).iloc[0]
        assert row["Precision"] == pytest.approx(1.0)
        assert row["Recall"] == pytest.approx(1.0)

    def test_threshold_of_one_predicts_nothing_positive_zero_division_handled(self):
        y_true = np.array([0, 1, 1])
        y_proba = np.array([0.1, 0.5, 0.6])
        row = threshold_sweep(y_true, y_proba, np.array([1.01])).iloc[0]
        assert row["Precision"] == 0.0 and row["Recall"] == 0.0

    def test_returns_one_row_per_threshold(self):
        thresholds = np.arange(0.1, 1.0, 0.1)
        result = threshold_sweep(np.array([0, 1]), np.array([0.3, 0.7]), thresholds)
        assert len(result) == len(thresholds)


class TestGroupWiseMetrics:
    def test_small_group_reports_nan_not_a_misleading_number(self):
        y_true = np.array([0, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.2])
        groups = pd.Series(["circuit_a"] * 3)
        result = group_wise_metrics(y_true, y_proba, groups, min_group_size=30)
        assert result["roc_auc"].isna().all()

    def test_large_well_separated_group_reports_high_roc_auc(self):
        rng = np.random.RandomState(0)
        n = 200
        y_true = rng.randint(0, 2, size=n)
        y_proba = np.where(y_true == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
        groups = pd.Series(["circuit_a"] * n)
        result = group_wise_metrics(y_true, y_proba, groups, min_group_size=30)
        assert result["roc_auc"].iloc[0] > 0.9

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            group_wise_metrics(np.array([0, 1]), np.array([0.1, 0.9, 0.5]), pd.Series(["a", "b"]))

    def test_groups_with_a_single_class_report_nan(self):
        """ROC-AUC is undefined with only one class present — must not silently compute a wrong number."""
        y_true = np.array([1] * 40)
        y_proba = np.random.RandomState(0).uniform(0, 1, 40)
        groups = pd.Series(["circuit_a"] * 40)
        result = group_wise_metrics(y_true, y_proba, groups, min_group_size=30)
        assert result["roc_auc"].isna().all()
