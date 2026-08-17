#!/usr/bin/env python3
"""
Ingest FastF1 race sessions into PostgreSQL.

Wires together the layers this project separates on purpose: fetch (raw,
untouched) -> clean (normalize + flag) -> write. This is the seam the prior
repo didn't have — see src/f1_pit_window/ingestion/fastf1_client.py's
docstring for why that mattered.

Usage:
    python scripts/ingest_data.py --years 2018,2019,2020,2021,2022,2023,2024 --races 1,2,3,...,24
    DATABASE_URL=postgresql://... python scripts/ingest_data.py --years 2024 --races 1,2
"""

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from f1_pit_window.data.cleaning import normalize_laps  # noqa: E402
from f1_pit_window.data.schema import Base, LapORM, RaceORM  # noqa: E402
from f1_pit_window.ingestion.fastf1_client import fetch_session  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default="2018,2019,2020,2021,2022,2023,2024")
    parser.add_argument("--races", default=",".join(str(i) for i in range(1, 25)))
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--cache-dir", default=".cache/")
    args = parser.parse_args()

    db_url = args.db_url or os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/f1_pit_db")
    logger.info("Connecting to %s", db_url.split("@")[-1])
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    years = [int(y) for y in args.years.split(",")]
    races = [int(r) for r in args.races.split(",")]

    stats = {"races": 0, "laps": 0, "errors": 0}
    for year in years:
        for race_num in races:
            try:
                raw, meta = fetch_session(year, race_num, cache_dir=args.cache_dir)
            except Exception as exc:
                logger.error("  ✗ %d R%d: %s", year, race_num, exc)
                stats["errors"] += 1
                continue
            if raw is None:
                stats["errors"] += 1
                continue

            cleaned = normalize_laps(raw, session_key=meta["session_key"])

            with Session(engine) as session:
                existing = session.query(RaceORM).filter_by(year=meta["year"], race_name=meta["race_name"]).first()
                if existing is None:
                    race = RaceORM(
                        year=meta["year"], race_name=meta["race_name"], circuit_name=meta["circuit_name"],
                        session_date=meta["session_date"], num_drivers=meta["num_drivers"], num_laps=meta["num_laps"],
                    )
                    session.add(race)
                    session.commit()
                    race_id = race.race_id
                else:
                    race_id = existing.race_id

                cleaned = cleaned.copy()
                cleaned["race_id"] = race_id
                columns = [c.name for c in LapORM.__table__.columns if c.name in cleaned.columns]
                session.bulk_insert_mappings(LapORM, cleaned[columns].to_dict("records"))
                session.commit()

            stats["races"] += 1
            stats["laps"] += len(cleaned)

    logger.info("Ingestion complete: %d races, %d laps, %d errors", stats["races"], stats["laps"], stats["errors"])
    return 0 if stats["races"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
