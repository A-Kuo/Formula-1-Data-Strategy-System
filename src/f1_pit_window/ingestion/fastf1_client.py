"""
FastF1 ingestion client
========================

Fetch-only. This module's job ends at "here is a raw FastF1 lap frame and its
race metadata" — it does not rename columns, convert units, or touch tyre
compound spellings. That's :mod:`f1_pit_window.data.cleaning`'s job.

This split isn't just organizational. The prior repo's ``scripts/ingest.py``
did fetching *and* normalization in one function, and its column-rename map
had the ``Time`` → ``lap_time_seconds`` bug (session-elapsed time, not lap
duration — see ``docs/decisions/``). Because fetching and normalizing were
one step, there was no seam where a test could exercise "did we normalize a
known-shape frame correctly" independent of "did we successfully hit the
FastF1 API" — the two concerns could only be tested together, over the
network, which is exactly why a wrong column mapping shipped unnoticed.
Here, :func:`fetch_session` returns FastF1's raw frame untouched; everything
past that point runs through :func:`f1_pit_window.data.cleaning.normalize_laps`,
which is tested against synthetic fixtures with no network involved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_session(year: int, race_num: int, cache_dir: str = ".cache/") -> tuple[pd.DataFrame | None, dict | None]:
    """
    Fetch one FastF1 race session's raw lap frame and race metadata.

    No normalization happens here — the returned frame has FastF1's native
    column names (``Driver``, ``LapTime``, ``Compound``, …), not the
    canonical contract. Pass it to
    :func:`f1_pit_window.data.cleaning.normalize_laps` next.

    Returns:
        ``(raw_laps, race_meta)``, or ``(None, None)`` if the session has no
        lap data (this is a normal outcome for e.g. a cancelled session, not
        an error).
    """
    import fastf1

    # fastf1.Cache.enable_cache() requires the directory to already exist —
    # on a fresh clone it doesn't, and the FastF1 error message for that
    # ("Cache directory does not exist!") is easy to mistake for a network
    # or credentials problem.
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    logger.info("Fetching %d race %d...", year, race_num)
    session = fastf1.get_session(year, race_num, "R")
    session.load(weather=True, telemetry=False)

    laps = session.laps
    if laps.empty:
        logger.warning("  No laps found for %d R%d", year, race_num)
        return None, None

    session_key = f"{session.event.year}-{session.event['RoundNumber']}"
    raw = laps.copy()
    raw["session_key"] = session_key
    if "Driver" in raw.columns:
        raw["driver_code_raw"] = raw["Driver"].apply(lambda x: getattr(x, "code", x) if x else None)
    if "Team" in raw.columns:
        raw["Team"] = raw["Team"].apply(lambda x: str(x) if x else None)

    race_meta = {
        "year": year,
        "race_name": session.event["EventName"],
        "circuit_name": session.event["Location"],
        "session_date": session.date if hasattr(session, "date") else None,
        "num_drivers": laps["Driver"].nunique(),
        "num_laps": laps["LapNumber"].max(),
        "session_key": session_key,
    }

    logger.info("  ✓ %d R%d: %d laps, %d drivers", year, race_num, len(raw), race_meta["num_drivers"])
    return raw, race_meta


def fetch_seasons(
    years: list[int],
    race_numbers: list[int],
    cache_dir: str = ".cache/",
) -> list[tuple[pd.DataFrame, dict]]:
    """
    Fetch every requested (year, race_num) combination that actually exists.

    Missing races (a season with fewer rounds than requested, a cancelled
    event) are skipped with a warning, not treated as a fatal error — a
    full-season pull should not abort because round 23 doesn't exist.
    """
    results: list[tuple[pd.DataFrame, dict]] = []
    for year in years:
        for race_num in race_numbers:
            try:
                raw, meta = fetch_session(year, race_num, cache_dir=cache_dir)
            except Exception as exc:
                logger.error("  ✗ %d R%d: %s", year, race_num, exc)
                continue
            if raw is not None:
                results.append((raw, meta))
    return results
