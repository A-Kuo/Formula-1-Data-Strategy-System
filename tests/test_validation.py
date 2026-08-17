"""Validation layer tests — completeness/consistency/reasonableness gates."""

import pandas as pd
import pytest

from f1_pit_window.data.validation import Status, ValidationConfig, validate_laps


@pytest.fixture
def clean_frame(laps_with_features):
    return laps_with_features


class TestCleanDataPasses:
    def test_fixture_passes_validation(self, clean_frame):
        report = validate_laps(clean_frame)
        assert report.passed, report.summary()

    def test_no_spurious_errors_on_well_formed_data(self, clean_frame):
        assert validate_laps(clean_frame).errors == []


class TestCompleteness:
    def test_missing_required_column_is_error(self, clean_frame):
        broken = clean_frame.drop(columns=["lap_time_seconds"])
        report = validate_laps(broken)
        assert not report.passed

    def test_null_rate_above_threshold_on_required_column_is_error(self, clean_frame):
        broken = clean_frame.copy()
        broken.loc[broken.sample(frac=0.5, random_state=1).index, "tyre_compound"] = None
        report = validate_laps(broken)
        assert not report.passed

    def test_too_few_drivers_is_error(self, clean_frame):
        single_driver = clean_frame[clean_frame["driver_number"] == 1]
        report = validate_laps(single_driver, ValidationConfig(min_drivers=2))
        assert not report.passed


class TestConsistency:
    def test_cumulative_gap_that_decreases_is_error(self, clean_frame):
        broken = clean_frame.copy()
        driver1_laps = broken[broken["driver_number"] == 1].sort_values("lap_number").index
        broken.loc[driver1_laps[len(driver1_laps) // 2], "gap_to_leader"] = -999.0
        assert not validate_laps(broken).passed

    def test_tyre_age_not_resetting_after_pit_is_error(self, clean_frame):
        broken = clean_frame.copy()
        out_lap = (broken["driver_number"] == 1) & (broken["lap_number"] == 11)
        broken.loc[out_lap, "tyre_age"] = 5
        assert not validate_laps(broken).passed

    def test_stint_change_reset_check_does_not_double_flag_the_lap_after_the_out_lap(self, clean_frame):
        """
        Regression test for the tyre_age_resets_after_pit bug: keying off
        is_pit_lap (true on both the in-lap and out-lap of a stop) demanded
        tyre_age == 0 one lap too late. Keying off stint_number instead
        means a clean fixture — which has real pit stops in it — must pass.
        """
        report = validate_laps(clean_frame)
        tyre_age_check = next(r for r in report.results if r.name == "tyre_age_resets_after_pit")
        assert tyre_age_check.status is Status.PASS

    def test_duplicate_lap_number_is_error(self, duplicate_lap_laps):
        from f1_pit_window.features.build_features import compute_all
        with_features = pd.concat(
            [duplicate_lap_laps.reset_index(drop=True), compute_all(duplicate_lap_laps).reset_index(drop=True)], axis=1,
        )
        assert not validate_laps(with_features).passed


class TestReasonableness:
    def test_lap_time_outside_bounds_is_error(self, clean_frame):
        broken = clean_frame.copy()
        broken.loc[broken.index[0], "lap_time_seconds"] = 5.0
        assert not validate_laps(broken).passed

    def test_degradation_rate_outside_bounds_is_error(self, clean_frame):
        broken = clean_frame.copy()
        broken.loc[broken.index[0], "degradation_rate"] = 5.0
        assert not validate_laps(broken).passed


class TestReportStructure:
    def test_to_json_is_valid_json(self, clean_frame):
        import json
        parsed = json.loads(validate_laps(clean_frame).to_json())
        assert parsed["n_rows"] == len(clean_frame)

    def test_summary_flags_passed_batch(self, clean_frame):
        assert validate_laps(clean_frame).summary().startswith("[PASS]")
