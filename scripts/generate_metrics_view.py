#!/usr/bin/env python3
"""
Regenerate the canonical metrics SQL view from f1_pit_window.features.build_features
========================================================================================

The SQL a dashboard or analyst would query is generated from the same
registry the Python feature pipeline uses, not maintained by hand — a
metric change shows up as a reviewable SQL diff in ``db/metrics_view.sql``,
not a silent edit to a shared view.

Usage:
    python scripts/generate_metrics_view.py                    # writes db/metrics_view.sql
    python scripts/generate_metrics_view.py --check             # CI mode: fail if stale
    python scripts/generate_metrics_view.py --as-of 2023-06-01  # pin to a past schema version
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from f1_pit_window.features.build_features import CANONICAL_REGISTRY, build_metrics_view_sql  # noqa: E402

DEFAULT_OUTPUT = _REPO_ROOT / "db" / "metrics_view.sql"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--as-of", type=str, default=None, help="Pin to definitions live on this date (YYYY-MM-DD)")
    parser.add_argument("--check", action="store_true", help="Don't write; exit non-zero if stale (CI gate)")
    args = parser.parse_args()

    when = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else None
    sql = build_metrics_view_sql(when=when)

    print(f"Registry fingerprint: {CANONICAL_REGISTRY.fingerprint()}", file=sys.stderr)
    print(f"Metrics: {', '.join(CANONICAL_REGISTRY.keys())}", file=sys.stderr)

    if args.check:
        if not args.output.exists():
            print(f"✗ {args.output} does not exist — run without --check to generate it", file=sys.stderr)
            return 1
        if args.output.read_text() != sql:
            print(f"✗ {args.output} is stale relative to build_features.py. Regenerate and commit.", file=sys.stderr)
            return 1
        print(f"✓ {args.output} is up to date", file=sys.stderr)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(sql)
    print(f"✓ Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
