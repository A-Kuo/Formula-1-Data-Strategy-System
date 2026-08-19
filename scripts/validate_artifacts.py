#!/usr/bin/env python3
"""
Validate freshly-trained model artifacts before they're published anywhere.

Thin CLI wrapper around the existing pre-serving gate,
f1_pit_window.modeling.inference.validate_model_artifacts — no new
validation logic lives here. Loads the model + held-out split
scripts/train_model.py just wrote, recomputes y_proba, and runs the same
check the Streamlit app runs before it will render a prediction. Intended
as the automated pass/fail gate in a scheduled retrain workflow: a bad run
exits non-zero and nothing downstream (release publish, benchmark commit)
happens.

Checks structural soundness only (no NaN/Inf, no out-of-range
probabilities, a well-formed metrics schema) — not whether the new model is
*better* than the previous one. Comparing against a prior release's metrics
is a deliberately deferred, real decision, not built here — see
docs/decisions/007-monolith-architecture.md's neighboring ADRs for how this
project records that kind of scope boundary explicitly rather than leaving
it implicit.

Usage:
    python scripts/validate_artifacts.py --model-dir models
"""

import argparse
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from f1_pit_window.modeling.inference import ModelArtifactValidationError, validate_model_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    try:
        with open(model_dir / "xgboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(model_dir / "metrics.pkl", "rb") as f:
            metrics = pickle.load(f)
        X_test = np.load(model_dir / "X_test_scaled.npy")
        y_test = np.load(model_dir / "y_test.npy")
    except FileNotFoundError as exc:
        print(f"✗ Missing artifact: {exc} — run scripts/train_model.py first", file=sys.stderr)
        return 1

    y_proba = model.predict_proba(X_test)[:, 1]

    try:
        validate_model_artifacts(X_test, y_test, y_proba, metrics)
    except ModelArtifactValidationError as exc:
        print(f"✗ Artifact validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"✓ Artifacts in {model_dir}/ passed validation "
          f"(ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
