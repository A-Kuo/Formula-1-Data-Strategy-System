"""Model-artifact validation gate tests — corrupted-artifact scenarios must be rejected."""

import numpy as np
import pytest

from f1_pit_window.modeling.inference import ModelArtifactValidationError, predict, validate_model_artifacts

GOOD_METRICS = {
    "roc_auc": 0.84, "f1": 0.49, "recall": 0.90, "precision": 0.34,
    "threshold": 0.60, "train_size": 16867, "test_size": 2801,
}


def _good_arrays(n=20):
    rng = np.random.RandomState(0)
    return rng.normal(size=(n, 4)), rng.randint(0, 2, size=n), rng.uniform(0, 1, size=n)


class TestValidateModelArtifacts:
    def test_good_artifacts_pass(self):
        validate_model_artifacts(*_good_arrays(), GOOD_METRICS)

    def test_nan_in_x_test_is_rejected(self):
        X, y, p = _good_arrays()
        X[0, 0] = np.nan
        with pytest.raises(ModelArtifactValidationError, match="X_test"):
            validate_model_artifacts(X, y, p, GOOD_METRICS)

    def test_non_binary_y_test_is_rejected(self):
        X, y, p = _good_arrays()
        y = y.astype(float)
        y[0] = 2
        with pytest.raises(ModelArtifactValidationError, match="binary"):
            validate_model_artifacts(X, y, p, GOOD_METRICS)

    def test_y_proba_above_one_is_rejected(self):
        X, y, p = _good_arrays()
        p[0] = 1.4
        with pytest.raises(ModelArtifactValidationError, match=r"\[0, 1\]"):
            validate_model_artifacts(X, y, p, GOOD_METRICS)

    def test_mismatched_lengths_are_rejected(self):
        X, y, p = _good_arrays()
        p = p[:-1]
        with pytest.raises(ModelArtifactValidationError, match="length mismatch"):
            validate_model_artifacts(X, y, p, GOOD_METRICS)

    def test_roc_auc_above_one_is_rejected(self):
        X, y, p = _good_arrays()
        with pytest.raises(ModelArtifactValidationError, match="saved_metrics"):
            validate_model_artifacts(X, y, p, {**GOOD_METRICS, "roc_auc": 1.4})

    def test_a_low_but_valid_roc_auc_is_not_rejected(self):
        X, y, p = _good_arrays()
        validate_model_artifacts(X, y, p, {**GOOD_METRICS, "roc_auc": 0.51, "f1": 0.02})


class TestPredict:
    def test_missing_feature_raises_keyerror(self):
        class DummyModel:
            def predict_proba(self, X):
                return np.array([[0.5, 0.5]])

        class DummyScaler:
            def transform(self, X):
                return X

        with pytest.raises(KeyError):
            predict(DummyModel(), DummyScaler(), {"a": 1.0}, ["a", "b"])

    def test_predict_returns_a_probability(self):
        class DummyModel:
            def predict_proba(self, X):
                return np.array([[0.3, 0.7]])

        class DummyScaler:
            def transform(self, X):
                return X

        result = predict(DummyModel(), DummyScaler(), {"a": 1.0, "b": 2.0}, ["a", "b"])
        assert result == pytest.approx(0.7)
