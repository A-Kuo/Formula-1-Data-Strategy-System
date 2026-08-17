"""
Feature registry tests: governance mechanics for the original four canonical
metrics, plus correctness tests for the two extensions added in this
project's re-scoping (formerly-ML-only features now governed; real
strategy-context features).
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from f1_pit_window.features.build_features import (
    CANONICAL_REGISTRY,
    MetricDefinition,
    MetricRegistry,
    build_circuit_pit_loss_table,
    build_metrics_view_sql,
    compute_all,
    compute_metric,
    opponent_pit_window_signal,
    pit_lane_loss_by_circuit,
)


class TestRegistryGovernance:
    def test_every_key_has_exactly_one_current_version(self):
        for key in CANONICAL_REGISTRY.keys():
            current = [v for v in CANONICAL_REGISTRY.versions(key) if CANONICAL_REGISTRY.get(key, v).is_current]
            assert len(current) == 1, f"{key} must have exactly one current version, found {current}"

    def test_extension_features_are_registered(self):
        for key in ("stint_age_squared", "race_progress", "pace_delta", "gap_closing_rate", "relative_pace_delta"):
            assert key in CANONICAL_REGISTRY.keys()

    def test_get_unknown_metric_raises_keyerror(self):
        with pytest.raises(KeyError):
            CANONICAL_REGISTRY.get("not_a_real_metric")

    def test_as_of_returns_v1_before_2024_and_v2_after(self):
        assert CANONICAL_REGISTRY.as_of(date(2023, 6, 1))["gap_to_leader"].version == "v1"
        assert CANONICAL_REGISTRY.as_of(date(2024, 6, 1))["gap_to_leader"].version == "v2"

    def test_extension_features_not_live_before_2024(self):
        """The new features didn't exist as governed metrics before this project's re-scoping."""
        live = CANONICAL_REGISTRY.as_of(date(2023, 6, 1))
        assert "gap_closing_rate" not in live

    def test_duplicate_version_registration_rejected(self):
        registry = MetricRegistry()
        definition = MetricDefinition(
            key="k", version="v1", title="t", unit="u", definition="d", rationale="r",
            owner="o", effective_from=date(2024, 1, 1), compute=lambda f: f["x"],
        )
        registry.register(definition)
        with pytest.raises(ValueError):
            registry.register(definition)


class TestOriginalMetricsUnchanged:
    """Spot-checks that the port preserved behavior — full correctness coverage lived in the prior repo's test suite."""

    def test_pit_delta_v2_reported_on_out_lap(self, laps_with_features):
        driver1 = laps_with_features[laps_with_features["driver_number"] == 1].sort_values("lap_number")
        result = compute_metric(driver1, "pit_delta", "v2")
        out_lap = result[driver1["lap_number"] == 11]
        assert out_lap.iloc[0] == pytest.approx(22.0)

    def test_tyre_age_v2_resets_to_zero_on_new_stint(self, laps_with_features):
        first_of_stint2 = laps_with_features[
            (laps_with_features["driver_number"] == 1) & (laps_with_features["lap_number"] == 11)
        ]
        assert compute_metric(first_of_stint2, "tyre_age", "v2").iloc[0] == 0


class TestExtension1FormerlyMlOnlyFeatures:
    def test_stint_age_squared_matches_tyre_age_squared(self, laps_with_features):
        tyre_age = compute_metric(laps_with_features, "tyre_age", "v2")
        expected = tyre_age.astype(float) ** 2
        result = compute_metric(laps_with_features, "stint_age_squared", "v1")
        np.testing.assert_allclose(result.values, expected.values)

    def test_race_progress_is_zero_to_one(self, laps_with_features):
        result = compute_metric(laps_with_features, "race_progress", "v1")
        assert result.between(0, 1).all()

    def test_race_progress_is_one_on_last_lap_of_session(self, laps_with_features):
        driver1 = laps_with_features[laps_with_features["driver_number"] == 1]
        last_lap = driver1[driver1["lap_number"] == driver1["lap_number"].max()]
        result = compute_metric(last_lap, "race_progress", "v1")
        assert result.iloc[0] == pytest.approx(1.0)

    def test_pace_delta_is_zero_for_a_constant_pace_stint(self):
        frame = pd.DataFrame({
            "session_key": ["S1"] * 5, "driver_number": [1] * 5,
            "lap_number": [1, 2, 3, 4, 5], "lap_time_seconds": [90.0] * 5,
        })
        result = compute_metric(frame, "pace_delta", "v1")
        np.testing.assert_allclose(result.values, 0.0)


class TestExtension2StrategyContextFeatures:
    def test_gap_closing_rate_positive_when_gap_shrinking(self):
        # Driver 1 always leads (gap=0); driver 2's gap to leader shrinks over time.
        frame = pd.DataFrame({
            "session_key": ["S1"] * 10,
            "driver_number": [1] * 5 + [2] * 5,
            "lap_number": [1, 2, 3, 4, 5] * 2,
            "lap_time_seconds": [90.0] * 5 + [94.0, 93.0, 92.0, 91.0, 90.0],
        })
        result = compute_metric(frame, "gap_closing_rate", "v1")
        driver2_last = result[(frame["driver_number"] == 2) & (frame["lap_number"] == 5)]
        assert driver2_last.iloc[0] > 0  # closing the gap

    def test_relative_pace_delta_zero_for_the_field_median_driver(self):
        frame = pd.DataFrame({
            "session_key": ["S1"] * 3, "driver_number": [1, 2, 3], "lap_number": [1, 1, 1],
            "lap_time_seconds": [89.0, 90.0, 91.0],
        })
        result = compute_metric(frame, "relative_pace_delta", "v1")
        assert result.iloc[1] == pytest.approx(0.0)  # driver 2 is exactly the median

    def test_circuit_pit_loss_table_requires_needed_columns(self):
        frame = pd.DataFrame({"circuit_name": ["Monza"]})
        with pytest.raises(KeyError):
            build_circuit_pit_loss_table(frame)

    def test_circuit_pit_loss_table_and_lookup_roundtrip(self):
        historical = pd.DataFrame({
            "circuit_name": ["Monaco", "Monaco", "Monza"],
            "pit_delta": [22.0, 24.0, 18.0],
        })
        table = build_circuit_pit_loss_table(historical)
        frame = pd.DataFrame({"circuit_name": ["Monaco", "Monza"]})
        result = pit_lane_loss_by_circuit(frame, table)
        assert result.iloc[0] == pytest.approx(23.0)
        assert result.iloc[1] == pytest.approx(18.0)

    def test_circuit_pit_loss_falls_back_to_global_mean_for_unknown_circuit(self):
        historical = pd.DataFrame({"circuit_name": ["Monaco", "Monza"], "pit_delta": [20.0, 30.0]})
        table = build_circuit_pit_loss_table(historical)
        frame = pd.DataFrame({"circuit_name": ["NewTrack"]})
        result = pit_lane_loss_by_circuit(frame, table)
        assert result.iloc[0] == pytest.approx(25.0)  # mean of [20, 30]

    def test_opponent_pit_window_signal_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            opponent_pit_window_signal(pd.DataFrame())


class TestComputeAll:
    def test_returns_bare_key_columns_for_every_current_metric(self, normalized_laps):
        result = compute_all(normalized_laps)
        assert set(result.columns) == set(CANONICAL_REGISTRY.keys())


class TestMetricsViewSql:
    def test_view_sql_excludes_pandas_only_extension_metrics(self):
        sql = build_metrics_view_sql()
        for key in ("stint_age_squared", "race_progress", "pace_delta", "gap_closing_rate", "relative_pace_delta"):
            assert f" AS {key}" not in sql

    def test_view_sql_includes_original_sql_backed_metrics(self):
        sql = build_metrics_view_sql()
        for key in ("gap_to_leader", "tyre_age", "pit_delta", "degradation_rate"):
            assert f" AS {key}" in sql
