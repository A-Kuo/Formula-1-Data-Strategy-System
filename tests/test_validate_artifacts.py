"""
scripts/validate_artifacts.py tests.

Invoked as a subprocess (like the CI workflow invokes it) rather than
imported, since scripts/ is a collection of standalone entry points, not an
importable package — matching this repo's existing convention of testing
business logic in src/f1_pit_window/ (validate_model_artifacts itself is
already covered by tests/test_inference.py) and treating scripts/ as thin,
subprocess-tested wiring around it.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "validate_artifacts.py"


def _write_valid_artifacts(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    X_train = rng.normal(size=(40, 3))
    y_train = (X_train[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(X_train, y_train)

    X_test = rng.normal(size=(10, 3))
    y_test = (X_test[:, 0] > 0).astype(int)
    metrics = {
        "roc_auc": 0.9, "f1": 0.8, "recall": 0.8, "precision": 0.8,
        "threshold": 0.5, "train_size": 40, "test_size": 10,
    }

    with open(model_dir / "xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(model_dir / "metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    np.save(model_dir / "X_test_scaled.npy", X_test)
    np.save(model_dir / "y_test.npy", y_test)


def _run(model_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--model-dir", str(model_dir)],
        capture_output=True, text=True,
    )


class TestValidateArtifacts:
    def test_valid_artifacts_pass_and_exit_zero(self, tmp_path):
        _write_valid_artifacts(tmp_path)
        result = _run(tmp_path)
        assert result.returncode == 0
        assert "passed validation" in result.stderr

    def test_missing_artifacts_directory_fails_with_actionable_message(self, tmp_path):
        result = _run(tmp_path / "does_not_exist")
        assert result.returncode == 1
        assert "run scripts/train_model.py first" in result.stderr

    def test_non_binary_y_test_fails_validation(self, tmp_path):
        _write_valid_artifacts(tmp_path)
        y_test = np.load(tmp_path / "y_test.npy")
        y_test = y_test.copy()
        y_test[0] = 2
        np.save(tmp_path / "y_test.npy", y_test)

        result = _run(tmp_path)

        assert result.returncode == 1
        assert "must be binary" in result.stderr

    def test_out_of_bounds_metric_fails_validation(self, tmp_path):
        _write_valid_artifacts(tmp_path)
        with open(tmp_path / "metrics.pkl", "rb") as f:
            metrics = pickle.load(f)
        metrics["roc_auc"] = 1.5
        with open(tmp_path / "metrics.pkl", "wb") as f:
            pickle.dump(metrics, f)

        result = _run(tmp_path)

        assert result.returncode == 1
        assert "failed schema validation" in result.stderr
