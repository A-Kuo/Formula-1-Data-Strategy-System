"""scripts/format_release_notes.py tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from format_release_notes import format_release_notes  # noqa: E402


class TestFormatReleaseNotes:
    def test_includes_all_headline_metrics(self):
        metrics = {
            "roc_auc": 0.8123, "f1": 0.7456, "precision": 0.71, "recall": 0.79,
            "train_size": 12345, "test_size": 678,
        }
        notes = format_release_notes(metrics)
        assert "0.8123" in notes
        assert "0.7456" in notes
        assert "0.7100" in notes
        assert "0.7900" in notes
        assert "12345" in notes
        assert "678" in notes

    def test_ignores_extra_metrics_keys(self):
        metrics = {
            "roc_auc": 0.5, "f1": 0.5, "precision": 0.5, "recall": 0.5,
            "train_size": 1, "test_size": 1, "cv_scores": {"XGBoost": 0.9}, "feature_cols": ["a", "b"],
        }
        notes = format_release_notes(metrics)
        assert "Scheduled retrain" in notes
