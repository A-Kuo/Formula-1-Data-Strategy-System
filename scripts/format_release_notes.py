#!/usr/bin/env python3
"""
Format models/metrics.pkl into a short release-notes markdown snippet.

Used by .github/workflows/scheduled-retrain.yml when publishing a
retrained model as a dated GitHub Release, so each release is
self-describing (headline metrics visible without downloading the
artifacts) rather than a bare file drop.

Usage:
    python scripts/format_release_notes.py --model-dir models > release_notes.md
"""

import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path


def format_release_notes(metrics: dict) -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    lines = [
        f"Scheduled retrain — {today}",
        "",
        f"- ROC-AUC: {metrics['roc_auc']:.4f}",
        f"- F1: {metrics['f1']:.4f}",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        f"- Train size: {metrics['train_size']}, Test size: {metrics['test_size']}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    with open(Path(args.model_dir) / "metrics.pkl", "rb") as f:
        metrics = pickle.load(f)

    print(format_release_notes(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
