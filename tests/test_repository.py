"""Repository (publish-if-valid gate) tests."""

import json

import pytest

from f1_pit_window.data.repository import load_snapshot, publish
from f1_pit_window.data.validation import ValidationConfig


class TestPublish:
    def test_valid_batch_is_published(self, laps_with_features, tmp_path):
        decision = publish(laps_with_features, tmp_path)
        assert decision.published
        assert decision.report.passed

    def test_invalid_batch_is_rejected_not_published(self, laps_with_features, tmp_path):
        broken = laps_with_features.drop(columns=["lap_time_seconds"])
        decision = publish(broken, tmp_path)
        assert not decision.published
        assert not (tmp_path / "current").exists()

    def test_rejected_batch_leaves_an_audit_trail(self, laps_with_features, tmp_path):
        broken = laps_with_features.drop(columns=["lap_time_seconds"])
        decision = publish(broken, tmp_path)
        payload = json.loads((decision.snapshot_path / "validation_report.json").read_text())
        assert payload["passed"] is False

    def test_dashboard_never_sees_a_rejected_batch(self, laps_with_features, tmp_path):
        good = laps_with_features
        publish(good, tmp_path, label="session-1")

        broken = laps_with_features.copy()
        broken.loc[broken.index[0], "lap_time_seconds"] = 5.0
        decision = publish(broken, tmp_path, label="session-2-corrupted")
        assert not decision.published

        served, meta = load_snapshot(tmp_path)
        assert meta["label"] == "session-1"

    def test_second_valid_batch_fully_replaces_the_first(self, laps_with_features, tmp_path):
        publish(laps_with_features, tmp_path, label="first")
        smaller = laps_with_features[laps_with_features["lap_number"] <= 5]
        decision = publish(smaller, tmp_path, label="second")
        assert decision.published
        served, meta = load_snapshot(tmp_path)
        assert meta["label"] == "second"
        assert len(served) == len(smaller)

    def test_custom_validation_config_is_honored(self, laps_with_features, tmp_path):
        strict = ValidationConfig(lap_time_bounds=(95.0, 96.0))
        decision = publish(laps_with_features, tmp_path, config=strict)
        assert not decision.published


class TestLoadSnapshot:
    def test_missing_snapshot_raises_filenotfounderror(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_snapshot(tmp_path)

    def test_loaded_frame_round_trips_columns(self, laps_with_features, tmp_path):
        publish(laps_with_features, tmp_path)
        served, _ = load_snapshot(tmp_path)
        assert set(served.columns) == set(laps_with_features.columns)
