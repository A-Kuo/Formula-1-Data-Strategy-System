"""Training pipeline tests: target computation and threshold tuning.

Full end-to-end training (candidate_models/cross_validate/prepare_training_split)
is exercised via the Makefile's `make train` target against real ingested
data, not in the unit suite — these tests cover the two pieces with a single
clear correctness contract that doesn't require a trained model to check.
"""

import numpy as np
import pandas as pd
import pytest

from f1_pit_window.modeling.train import (
    DEFAULT_FEATURE_COLS,
    candidate_models,
    compute_target,
    cross_validate,
    prepare_training_split,
    tune_threshold,
)


class TestComputeTarget:
    def test_lap_immediately_before_a_pit_is_positive(self):
        frame = pd.DataFrame({
            "session_key": ["S1"] * 6, "driver_number": [1] * 6,
            "lap_number": [1, 2, 3, 4, 5, 6],
            "is_pit_lap": [False, False, False, False, True, False],
        })
        target = compute_target(frame, lookahead_laps=5)
        assert target.iloc[3] == 1  # lap 4, pit is on lap 5 (within next 5 laps)

    def test_lap_far_from_any_pit_is_negative(self):
        frame = pd.DataFrame({
            "session_key": ["S1"] * 10, "driver_number": [1] * 10,
            "lap_number": list(range(1, 11)),
            "is_pit_lap": [False] * 8 + [True, False],
        })
        target = compute_target(frame, lookahead_laps=5)
        assert target.iloc[0] == 0  # lap 1, pit is on lap 9 — outside the 5-lap window

    def test_does_not_leak_across_driver_boundary(self):
        frame = pd.DataFrame({
            "session_key": ["S1"] * 4, "driver_number": [1, 1, 2, 2],
            "lap_number": [1, 2, 1, 2],
            "is_pit_lap": [False, False, False, True],
        })
        target = compute_target(frame, lookahead_laps=5)
        # Driver 1's lap 1 must not see driver 2's pit stop.
        assert target.iloc[0] == 0

    def test_requires_is_pit_lap_column(self):
        with pytest.raises(KeyError):
            compute_target(pd.DataFrame({"session_key": ["S1"], "driver_number": [1], "lap_number": [1]}))

    def test_last_laps_of_a_session_with_no_more_pits_are_negative(self):
        frame = pd.DataFrame({
            "session_key": ["S1"] * 3, "driver_number": [1] * 3,
            "lap_number": [1, 2, 3], "is_pit_lap": [False, False, False],
        })
        target = compute_target(frame, lookahead_laps=5)
        assert (target == 0).all()


class TestTrainingPipelineIntegration:
    """
    Exercises prepare_training_split / candidate_models / cross_validate
    together on a synthetic-but-realistically-shaped dataset — not asserting
    on specific accuracy numbers (that would make the test flaky and
    wouldn't mean much on synthetic data), only that the pipeline runs
    end-to-end and produces internally consistent shapes.
    """

    @pytest.fixture(scope="class")
    def multi_session_normalized(self):
        from f1_pit_window.data.cleaning import normalize_laps

        rng = np.random.RandomState(11)
        frames = []
        for session_idx in range(6):
            year = 2018 if session_idx < 5 else 2024
            rows = []
            for driver in range(1, 5):
                pit_lap = rng.randint(8, 20)
                for lap in range(1, 26):
                    rows.append({
                        "DriverNumber": str(driver), "Driver": f"D{driver:02d}", "Team": "Team A",
                        "LapNumber": lap, "LapTime": pd.to_timedelta(90 + rng.normal(0, 1), unit="s"),
                        "Compound": "SOFT", "TyreLife": lap % (pit_lap + 1), "Stint": 1 + int(lap > pit_lap),
                        "Position": driver,
                        "PitInTime": pd.to_timedelta(500, unit="s") if lap == pit_lap else pd.NaT,
                        "PitOutTime": pd.to_timedelta(520, unit="s") if lap == pit_lap + 1 else pd.NaT,
                        "AirTemp": 25.0, "TrackTemp": 35.0, "WindSpeed": 5.0, "Humidity": 50.0, "Rainfall": False,
                    })
            raw = pd.DataFrame(rows)
            normalized = normalize_laps(raw, session_key=f"{year}-{session_idx}")
            normalized["year"] = year
            frames.append(normalized)
        return pd.concat(frames, ignore_index=True)

    def test_prepare_training_split_produces_consistent_shapes(self, multi_session_normalized):
        split = prepare_training_split(multi_session_normalized, train_years=[2018], test_years=[2024])
        assert split.X_train.shape[1] == len(DEFAULT_FEATURE_COLS)
        assert split.X_train.shape[0] == len(split.y_train)
        assert split.X_test.shape[0] == len(split.y_test)
        assert split.X_train.shape[0] > 0 and split.X_test.shape[0] > 0

    def test_candidate_models_returns_the_three_model_comparison(self, multi_session_normalized):
        split = prepare_training_split(multi_session_normalized, train_years=[2018], test_years=[2024])
        models = candidate_models(split.y_train)
        assert set(models.keys()) == {"Logistic Regression", "Random Forest", "XGBoost"}

    def test_cross_validate_returns_a_score_per_model(self, multi_session_normalized):
        split = prepare_training_split(multi_session_normalized, train_years=[2018], test_years=[2024])
        models = candidate_models(split.y_train)
        scores = cross_validate(models, split.X_train, split.y_train, n_splits=3)
        assert set(scores.keys()) == set(models.keys())
        assert all(0.0 <= score <= 1.0 for score in scores.values())


class TestTuneThreshold:
    def test_finds_a_reasonable_threshold_for_well_separated_classes(self):
        y_true = np.array([0] * 50 + [1] * 50)
        y_proba = np.concatenate([
            np.random.RandomState(0).uniform(0, 0.3, 50),
            np.random.RandomState(1).uniform(0.7, 1.0, 50),
        ])
        tau = tune_threshold(y_true, y_proba)
        # tune_threshold grid-searches np.arange(0.1, 0.9, 0.01), whose float
        # accumulation can land a hair under a "round" boundary (0.29999999999999993
        # instead of 0.3) — a tolerance, not a stricter equality, is correct here.
        assert 0.3 - 1e-6 <= tau <= 0.7

    def test_returns_a_value_in_the_searched_range(self):
        y_true = np.random.RandomState(0).randint(0, 2, 100)
        y_proba = np.random.RandomState(1).uniform(0, 1, 100)
        tau = tune_threshold(y_true, y_proba)
        assert 0.1 <= tau < 0.9
