"""
Data contract
=============

The schema every downstream layer — cleaning, features, modeling, the
dashboard — is allowed to assume. This module declares the contract; it does
not enforce it (that's :mod:`f1_pit_window.data.cleaning`) or check it against
a batch (that's :mod:`f1_pit_window.data.validation`). Keeping the *what* and
the *how* in separate files is deliberate: this file should be readable by
someone who has never seen FastF1's raw column names.

Ported from ``f1pipeline/normalize.py`` (module-level constants only — see
``cleaning.py`` for the transformation functions, and
``MIGRATION_MAP.md`` for what changed structurally in the port).
"""

from __future__ import annotations

from enum import Enum


class TyreCompound(str, Enum):
    """
    Canonical tyre compound vocabulary.

    Pirelli ran a five-tier dry range in 2018 (HYPERSOFT … SUPERHARD) and
    switched to the three-tier SOFT/MEDIUM/HARD naming in 2019. Strategy
    compares stints *across* seasons, so the contract uses the three-tier
    form and maps the 2018 names onto it.

    Trade-off, stated explicitly because it is lossy: a 2018 HYPERSOFT and a
    2018 SUPERSOFT both normalize to ``SOFT``. Analyses that need the 2018
    five-tier granularity must read ``tyre_compound_raw``, which
    :func:`f1_pit_window.data.cleaning.normalize_laps` preserves.
    """

    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"
    UNKNOWN = "UNKNOWN"


#: Raw compound spellings seen in FastF1 2018–2024 → canonical compound.
COMPOUND_ALIASES: dict[str, TyreCompound] = {
    "HYPERSOFT": TyreCompound.SOFT,
    "ULTRASOFT": TyreCompound.SOFT,
    "SUPERSOFT": TyreCompound.SOFT,
    "SOFT": TyreCompound.SOFT,
    "S": TyreCompound.SOFT,
    "MEDIUM": TyreCompound.MEDIUM,
    "M": TyreCompound.MEDIUM,
    "HARD": TyreCompound.HARD,
    "SUPERHARD": TyreCompound.HARD,
    "H": TyreCompound.HARD,
    "INTERMEDIATE": TyreCompound.INTERMEDIATE,
    "INTER": TyreCompound.INTERMEDIATE,
    "I": TyreCompound.INTERMEDIATE,
    "WET": TyreCompound.WET,
    "W": TyreCompound.WET,
    "WETS": TyreCompound.WET,
    "WET WEATHER": TyreCompound.WET,
    "TEST_UNKNOWN": TyreCompound.UNKNOWN,
    "UNKNOWN": TyreCompound.UNKNOWN,
    "NAN": TyreCompound.UNKNOWN,
}

#: FastF1 lap-frame column → canonical column name.
#:
#: Two mappings here are load-bearing and were wrong in the ingest script this
#: project originally shipped with (``scripts/ingest.py``, prior repo):
#:
#: * ``LapTime`` is the lap duration. ``Time`` is *session elapsed time* at the
#:   end of the lap — monotonically increasing across the race. Mapping ``Time``
#:   to ``lap_time_seconds`` yields lap times that climb from ~90s to ~5000s.
#:   See ``docs/decisions/`` for the full account of that bug.
#: * ``Driver`` is the three-letter code ("VER"); ``DriverNumber`` is the car
#:   number. They are not interchangeable.
FASTF1_COLUMN_MAP: dict[str, str] = {
    "LapTime": "lap_time_seconds",
    "DriverNumber": "driver_number",
    "Driver": "driver_code",
    "Team": "team_name",
    "LapNumber": "lap_number",
    "Compound": "tyre_compound",
    "TyreLife": "tyre_life",
    "Stint": "stint_number",
    "Position": "position",
    "Sector1Time": "sector_1",
    "Sector2Time": "sector_2",
    "Sector3Time": "sector_3",
    "PitInTime": "pit_in_time",
    "PitOutTime": "pit_out_time",
    "AirTemp": "air_temp",
    "TrackTemp": "track_temp",
    "WindSpeed": "wind_speed",
    "Humidity": "humidity",
    "Rainfall": "rainfall",
    "TrackStatus": "track_status",
}

#: Canonical columns that hold a duration in seconds.
DURATION_COLUMNS: tuple[str, ...] = (
    "lap_time_seconds",
    "sector_1",
    "sector_2",
    "sector_3",
    "pit_in_time",
    "pit_out_time",
    "pit_loss_seconds",
    "gap_to_leader",
    "gap_to_car_in_front",
)

#: Canonical column names the rest of the pipeline may rely on existing.
#: This IS the data contract — anything not listed here is not guaranteed.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "session_key",
    "driver_number",
    "driver_code",
    "team_name",
    "lap_number",
    "lap_time_seconds",
    "sector_1",
    "sector_2",
    "sector_3",
    "is_pit_lap",
    "pit_loss_seconds",
    "tyre_compound",
    "tyre_life",
    "stint_number",
    "position",
    "gap_to_leader",
    "gap_to_car_in_front",
    "air_temp",
    "track_temp",
    "wind_speed",
    "humidity",
    "rainfall",
    "circuit_name",
)
