"""
Repository: the publish-if-valid / serve-last-known-good gate
================================================================

Ported from ``f1pipeline/gate.py``, renamed to match this project's
"repository" naming (this is the persistence boundary the rest of the app
reads through). Behavior unchanged: a batch only reaches
:func:`load_snapshot` if it passed :func:`f1_pit_window.data.validation.validate_laps`
first. A failing batch is rejected with an audit trail; the previous
snapshot keeps serving.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from f1_pit_window.data.validation import ValidationConfig, ValidationReport, validate_laps

logger = logging.getLogger(__name__)

CURRENT_DIRNAME = "current"
REPORT_FILENAME = "validation_report.json"
DATA_FILENAME = "laps.parquet"
META_FILENAME = "meta.json"


@dataclass(frozen=True)
class GateDecision:
    """The outcome of a publish attempt: whether the batch was published, why, its validation report, and the
    snapshot path if it passed the gate."""

    published: bool
    reason: str
    report: ValidationReport
    snapshot_path: Path | None = None

    def to_dict(self) -> dict:
        return {
            "published": self.published, "reason": self.reason,
            "snapshot_path": str(self.snapshot_path) if self.snapshot_path else None,
            "report": self.report.to_dict(),
        }


def publish(
    frame: pd.DataFrame,
    snapshot_dir: str | Path,
    config: ValidationConfig | None = None,
    label: str | None = None,
) -> GateDecision:
    """
    Validate a batch and, only if it passes, promote it to the current snapshot.

    The write is atomic with respect to readers — built in a sibling temp
    directory and swapped into place — so :func:`load_snapshot` never observes
    a half-written directory. A failing batch is still recorded, under
    ``<snapshot_dir>/rejected/<timestamp>/``, so a validation failure leaves an
    audit trail even though it never reaches ``current/``.
    """
    root = Path(snapshot_dir)
    report = validate_laps(frame, config)
    now = datetime.now(timezone.utc)

    if not report.passed:
        rejected_dir = root / "rejected" / now.strftime("%Y%m%dT%H%M%SZ")
        rejected_dir.mkdir(parents=True, exist_ok=True)
        (rejected_dir / REPORT_FILENAME).write_text(report.to_json())
        reason = f"validation failed: {len(report.errors)} error(s) — see report"
        logger.error("Publish rejected (%s): %s", rejected_dir, reason)
        return GateDecision(published=False, reason=reason, report=report, snapshot_path=rejected_dir)

    current_dir = root / CURRENT_DIRNAME
    with tempfile.TemporaryDirectory(dir=_ensure_dir(root)) as tmp:
        staging = Path(tmp) / "staging"
        staging.mkdir()
        frame.to_parquet(staging / DATA_FILENAME, index=False)
        (staging / REPORT_FILENAME).write_text(report.to_json())
        meta = {"published_at": now.isoformat(), "label": label, "n_rows": len(frame), "stale": False}
        (staging / META_FILENAME).write_text(json.dumps(meta, indent=2))

        if current_dir.exists():
            backup = Path(tmp) / "previous"
            shutil.move(str(current_dir), str(backup))
        shutil.move(str(staging), str(current_dir))

    logger.info("Published %d rows to %s", len(frame), current_dir)
    return GateDecision(published=True, reason="validation passed", report=report, snapshot_path=current_dir)


def load_snapshot(snapshot_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """
    Read the current published snapshot — by construction, the last batch
    that passed the gate. Callers (the dashboard, `modeling/inference.py`)
    should always read through this function, never around it.

    Raises:
        FileNotFoundError: If no snapshot has ever been published.
    """
    current_dir = Path(snapshot_dir) / CURRENT_DIRNAME
    data_path = current_dir / DATA_FILENAME
    meta_path = current_dir / META_FILENAME
    if not data_path.exists():
        raise FileNotFoundError(
            f"No published snapshot at {current_dir}. Run publish() with a batch that passes validation first."
        )
    frame = pd.read_parquet(data_path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return frame, meta


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
