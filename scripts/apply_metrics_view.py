#!/usr/bin/env python3
"""
Apply db/metrics_view.sql to a live database.

Separate from `make view` (which only regenerates the SQL *file* from the
feature registry) because applying it requires the `laps` table to already
exist — which happens at ingest time
(scripts/ingest_data.py's Base.metadata.create_all), not at database
container boot. Run this after the first successful ingest, and again any
time db/metrics_view.sql changes.

Usage:
    DATABASE_URL=postgresql://... python scripts/apply_metrics_view.py
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sqlalchemy import create_engine  # noqa: E402


def main() -> int:
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/f1_pit_db")
    view_path = _REPO_ROOT / "db" / "metrics_view.sql"
    if not view_path.exists():
        print(f"✗ {view_path} does not exist — run `make view` first", file=sys.stderr)
        return 1

    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.exec_driver_sql(view_path.read_text())
    print(f"✓ Applied {view_path} to {db_url.split('@')[-1]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
