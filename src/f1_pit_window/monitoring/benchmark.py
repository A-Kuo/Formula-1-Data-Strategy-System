"""
Pipeline benchmark recording
=============================

Times pipeline stages (data load, feature computation, training) and
appends rows-processed/duration/throughput to a git-tracked CSV — a
lightweight record of pipeline performance over time and commits.
Deliberately separate from model-quality metrics (``models/metrics.pkl``,
produced by ``scripts/train_model.py``): this module records only runtime
and data volume, never accuracy/ROC-AUC/etc., so there is exactly one place
a model-quality number can come from and exactly one place a throughput
number can come from.
"""

from __future__ import annotations

import csv
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

CSV_COLUMNS = (
    "run_id", "timestamp_utc", "git_commit", "stage",
    "rows_processed", "duration_seconds", "rows_per_second", "notes",
)


def _git_commit() -> str:
    """Short commit SHA for correlating a benchmark row with the code that produced it."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


@dataclass
class BenchmarkRun:
    """One `make benchmark` invocation — a shared run_id/git_commit across every stage it records."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    git_commit: str = field(default_factory=_git_commit)
    rows: list[dict] = field(default_factory=list)

    def record(self, stage: str, rows_processed: int, duration_seconds: float, notes: str = "") -> None:
        rows_per_second = rows_processed / duration_seconds if duration_seconds > 0 else 0.0
        self.rows.append({
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": self.git_commit,
            "stage": stage,
            "rows_processed": rows_processed,
            "duration_seconds": round(duration_seconds, 3),
            "rows_per_second": round(rows_per_second, 1),
            "notes": notes,
        })


def write_results(run: BenchmarkRun, output_path: Path) -> None:
    """Append `run`'s rows to `output_path`, writing the header only if the file is new or empty."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_has_content = output_path.exists() and output_path.stat().st_size > 0
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_has_content:
            writer.writeheader()
        for row in run.rows:
            writer.writerow(row)
