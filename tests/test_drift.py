"""Drift monitoring tests — Population Stability Index."""

import numpy as np
import pandas as pd
import pytest

from f1_pit_window.monitoring.drift import feature_drift_report, population_stability_index, severity


class TestPopulationStabilityIndex:
    def test_identical_distributions_score_near_zero(self):
        rng = np.random.RandomState(0)
        reference = rng.normal(0, 1, 5000)
        current = rng.normal(0, 1, 5000)
        assert population_stability_index(reference, current) < 0.05

    def test_shifted_distribution_scores_high(self):
        rng = np.random.RandomState(0)
        reference = rng.normal(0, 1, 5000)
        current = rng.normal(5, 1, 5000)  # completely disjoint from reference
        assert population_stability_index(reference, current) > 0.25

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            population_stability_index(np.array([]), np.array([1.0, 2.0]))

    def test_zero_variance_reference_returns_zero_not_a_crash(self):
        reference = np.full(100, 5.0)
        current = np.full(100, 5.0)
        assert population_stability_index(reference, current) == 0.0

    def test_nan_values_are_dropped_not_propagated(self):
        rng = np.random.RandomState(0)
        reference = rng.normal(0, 1, 1000)
        current = np.concatenate([rng.normal(0, 1, 1000), [np.nan, np.nan]])
        result = population_stability_index(reference, current)
        assert np.isfinite(result)


class TestSeverity:
    def test_bands(self):
        assert severity(0.05) == "stable"
        assert severity(0.15) == "moderate"
        assert severity(0.30) == "significant"


class TestFeatureDriftReport:
    def test_reports_one_row_per_feature(self):
        rng = np.random.RandomState(0)
        reference = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500)})
        current = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(3, 1, 500)})
        report = feature_drift_report(reference, current, ["a", "b"])
        assert set(report["feature"]) == {"a", "b"}

    def test_shifted_feature_ranks_above_stable_feature(self):
        rng = np.random.RandomState(0)
        reference = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500)})
        current = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(4, 1, 500)})
        report = feature_drift_report(reference, current, ["a", "b"])
        assert report.iloc[0]["feature"] == "b"

    def test_missing_column_reported_as_unavailable_not_raised(self):
        reference = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        current = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        report = feature_drift_report(reference, current, ["a", "nonexistent"])
        row = report[report["feature"] == "nonexistent"].iloc[0]
        assert "unavailable" in row["severity"]
