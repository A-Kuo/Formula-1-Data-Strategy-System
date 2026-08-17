"""
Cleaning layer tests (normalization + imputation).

Each test targets a specific bug this layer exists to close: the ``Time``
vs ``LapTime`` column confusion, the 2018 five-tier compound vocabulary,
unflagged sensor spikes, and unbounded/zero-filled sensor gaps.
"""

import numpy as np
import pandas as pd
import pytest

from f1_pit_window.data.cleaning import (
    FORWARD_FILLED,
    SENSOR_FAILURE,
    flag_outliers,
    impute_gaps,
    normalize_compound,
    normalize_laps,
    to_seconds,
)
from f1_pit_window.data.contracts import TyreCompound


class TestToSeconds:
    def test_timedelta(self):
        assert to_seconds(pd.Timedelta(seconds=92.5)) == pytest.approx(92.5)

    def test_clock_string_with_minutes(self):
        assert to_seconds("1:32.456") == pytest.approx(92.456)

    def test_plain_number_read_as_seconds_by_default(self):
        assert to_seconds(92.5) == pytest.approx(92.5)

    def test_milliseconds_unit(self):
        assert to_seconds(92500, unit="ms") == pytest.approx(92.5)

    def test_none_is_nan(self):
        assert np.isnan(to_seconds(None))

    def test_unparseable_string_is_nan_not_raise(self):
        assert np.isnan(to_seconds("not-a-time"))

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError):
            to_seconds(1.0, unit="fortnights")


class TestNormalizeCompound:
    @pytest.mark.parametrize("raw,expected", [
        ("HYPERSOFT", TyreCompound.SOFT), ("SUPERSOFT", TyreCompound.SOFT),
        ("Medium", TyreCompound.MEDIUM), ("HARD", TyreCompound.HARD),
        ("intermediate", TyreCompound.INTERMEDIATE), ("wet", TyreCompound.WET),
    ])
    def test_known_aliases(self, raw, expected):
        assert normalize_compound(raw) == expected

    def test_2018_five_tier_and_2019_three_tier_collapse_to_same_bucket(self):
        assert normalize_compound("HYPERSOFT") == normalize_compound("SOFT") == TyreCompound.SOFT

    def test_none_is_unknown(self):
        assert normalize_compound(None) is TyreCompound.UNKNOWN

    def test_unrecognised_value_is_unknown_not_raise(self):
        assert normalize_compound("GLITTER") is TyreCompound.UNKNOWN


class TestFlagOutliers:
    def test_flags_the_planted_safety_car_lap(self):
        frame = pd.DataFrame({
            "session_key": ["S1"] * 10, "driver_number": [1] * 10,
            "lap_time_seconds": [92.0, 92.1, 92.3, 92.2, 125.0, 92.4, 92.1, 92.2, 92.3, 92.0],
        })
        flags = flag_outliers(frame, "lap_time_seconds")
        assert flags.iloc[4]
        assert not flags.iloc[0]

    def test_small_group_never_flagged(self):
        frame = pd.DataFrame({
            "session_key": ["S1"] * 3, "driver_number": [1] * 3, "lap_time_seconds": [92.0, 92.0, 500.0],
        })
        flags = flag_outliers(frame, "lap_time_seconds", min_group_size=5)
        assert not flags.any()

    def test_outlier_status_is_per_group_not_global(self):
        frame = pd.DataFrame({
            "session_key": ["S1"] * 12, "driver_number": [1] * 6 + [2] * 6,
            "lap_time_seconds": [92.0, 92.1, 92.2, 92.0, 92.1, 92.2] + [98.0, 98.1, 98.2, 98.0, 98.1, 98.2],
        })
        flags = flag_outliers(frame, "lap_time_seconds")
        assert not flags.any()


class TestNormalizeLaps:
    def test_lap_time_uses_laptime_column_not_session_elapsed_time(self, synthetic_laps):
        result = normalize_laps(synthetic_laps, session_key="2024-01")
        assert result["lap_time_seconds"].between(80, 130).all()

    def test_driver_number_and_driver_code_are_not_swapped(self, synthetic_laps):
        result = normalize_laps(synthetic_laps, session_key="2024-01")
        assert set(result["driver_number"].unique()) == {1, 2}
        assert set(result["driver_code"].unique()) == {"D01", "D02"}

    def test_compound_normalized_and_raw_preserved(self, synthetic_laps):
        result = normalize_laps(synthetic_laps, session_key="2024-01")
        driver1 = result[result["driver_number"] == 1]
        assert (driver1["tyre_compound"] == "SOFT").all()
        assert (driver1["tyre_compound_raw"] == "HYPERSOFT").all()

    def test_is_pit_lap_derived_from_pit_timestamps(self, normalized_laps):
        pit_laps = normalized_laps[normalized_laps["is_pit_lap"]]
        assert set(zip(pit_laps["driver_number"], pit_laps["lap_number"])) == {(1, 10), (1, 11), (2, 12), (2, 13)}

    def test_planted_outlier_is_flagged(self, normalized_laps):
        flagged = normalized_laps[(normalized_laps["driver_number"] == 1) & (normalized_laps["lap_number"] == 5)]
        assert flagged["lap_time_outlier"].iloc[0]

    def test_missing_values_are_not_zero_filled(self, synthetic_laps):
        result = normalize_laps(synthetic_laps, session_key="2024-01")
        missing = result[(result["driver_number"] == 2) & (result["lap_number"].isin([8, 9]))]
        assert missing["track_temp"].isna().all()

    def test_does_not_mutate_input(self, synthetic_laps):
        before = synthetic_laps.copy(deep=True)
        normalize_laps(synthetic_laps, session_key="2024-01")
        pd.testing.assert_frame_equal(synthetic_laps, before)

    def test_missing_session_key_without_default_raises(self):
        frame = pd.DataFrame({"LapNumber": [1], "LapTime": [pd.Timedelta(seconds=90)]})
        with pytest.raises(ValueError):
            normalize_laps(frame)


def _trace(values, interval=1.0):
    return pd.DataFrame({
        "session_key": ["S1"] * len(values), "driver_number": [1] * len(values),
        "t": np.arange(len(values)) * interval, "track_temp": values,
    })


class TestImputeGaps:
    def test_short_gap_is_forward_filled_and_marked(self):
        frame = _trace([41.0, 41.2, np.nan, 41.6, 41.8])
        result, report = impute_gaps(frame, ["track_temp"], time_col="t", max_gap_seconds=1.0)
        assert result["track_temp"].iloc[2] == pytest.approx(41.2)
        assert result["track_temp_imputed"].iloc[2]
        assert report.filled_samples == 1
        assert report.events[0].action == FORWARD_FILLED

    def test_long_gap_is_left_missing_and_marked_as_failure(self):
        frame = _trace([41.0, np.nan, np.nan, np.nan, 41.8])
        result, report = impute_gaps(frame, ["track_temp"], time_col="t", max_gap_seconds=1.0)
        assert result["track_temp"].iloc[1:4].isna().all()
        assert result["track_temp_sensor_failure"].iloc[1:4].all()
        assert report.failed_samples == 3
        assert report.events[0].action == SENSOR_FAILURE

    def test_gap_at_start_of_trace_is_always_sensor_failure(self):
        frame = _trace([np.nan, 41.2, 41.4])
        result, report = impute_gaps(frame, ["track_temp"], time_col="t", max_gap_seconds=100.0)
        assert result["track_temp_sensor_failure"].iloc[0]

    def test_no_time_col_or_interval_treated_as_failure(self):
        frame = _trace([41.0, np.nan, 41.5])
        result, report = impute_gaps(frame, ["track_temp"], max_gap_seconds=1.0)
        assert result["track_temp_sensor_failure"].iloc[1]

    def test_negative_max_gap_raises(self):
        frame = _trace([41.0, np.nan])
        with pytest.raises(ValueError):
            impute_gaps(frame, ["track_temp"], time_col="t", max_gap_seconds=-1.0)

    def test_fixture_short_gap_filled(self, normalized_laps):
        result, report = impute_gaps(
            normalized_laps, ["track_temp"], sample_interval_seconds=92.0, max_gap_seconds=200.0,
        )
        gap_rows = result[(result["driver_number"] == 2) & (result["lap_number"].isin([8, 9]))]
        assert gap_rows["track_temp_imputed"].all()
        assert report.filled_samples == 2
