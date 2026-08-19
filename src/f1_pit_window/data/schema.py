"""
Database schema
================

**Consolidated from two competing, incompatible ORM definitions** in the
prior repo: ``scripts/ingest.py`` (29-column raw-telemetry ``laps`` table —
this one) and ``sql_utils.py`` (a *different* schema with pre-computed
feature columns baked directly into ``laps`` — ``degradation_rate``,
``gap_to_leader``, etc. as table columns rather than a queryable view). The
two were never reconciled; nothing in the prior repo enforced that they
agreed, because nothing ever ran both against the same database at once.
That's the same "two schemas nobody reconciled" failure mode as the
duplicate ``DegradationRate`` implementations documented in
``docs/decisions/`` — just at the database layer instead of the feature
layer.

This module keeps ``scripts/ingest.py``'s shape: raw telemetry in ``laps``,
computed metrics served through the generated
``canonical_lap_metrics`` SQL view
(:func:`f1_pit_window.features.build_features.build_metrics_view_sql`), not
baked into the table. ``sql_utils.py``'s competing ORM is not ported — see
``MIGRATION_MAP.md``.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class RaceORM(Base):
    """F1 race metadata."""

    __tablename__ = "races"

    race_id = Column(Integer, primary_key=True)
    year = Column(Integer, nullable=False)
    race_name = Column(String(100), nullable=False)
    circuit_name = Column(String(100))
    session_date = Column(DateTime)
    num_drivers = Column(Integer)
    num_laps = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Race {self.year} {self.race_name}>"


class LapORM(Base):
    """
    Raw (normalized-but-not-feature-engineered) lap telemetry.

    Deliberately holds only what :mod:`f1_pit_window.data.cleaning` produces
    — identifiers, lap performance, tyre/stint, position/gaps, weather,
    incident flags. Computed metrics (gap_to_leader, tyre_age, pit_delta,
    degradation_rate, and the strategy-context features) are never written
    here; they're served live through the generated SQL view so there is
    exactly one place their definition can live.
    """

    __tablename__ = "laps"

    lap_id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Integer, ForeignKey("races.race_id"), nullable=False)
    session_key = Column(String(50), nullable=False)
    driver_number = Column(Integer, nullable=False)
    driver_code = Column(String(3))
    team_name = Column(String(50))
    lap_number = Column(Integer, nullable=False)

    lap_time_seconds = Column(Float)
    sector_1 = Column(Float)
    sector_2 = Column(Float)
    sector_3 = Column(Float)
    is_pit_lap = Column(Boolean, default=False)
    pit_in_time = Column(Float)
    pit_out_time = Column(Float)

    tyre_compound = Column(String(20))
    tyre_compound_raw = Column(String(20))
    tyre_life = Column(Integer)
    stint_number = Column(Integer)

    position = Column(Integer)
    position_text = Column(String(10))

    air_temp = Column(Float)
    track_temp = Column(Float)
    wind_speed = Column(Float)
    humidity = Column(Float)
    rainfall = Column(Boolean)

    is_safety_car_lap = Column(Boolean, default=False)
    is_yellow_flag = Column(Boolean, default=False)
    is_red_flag = Column(Boolean, default=False)
    lap_time_outlier = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Lap {self.session_key} D{self.driver_number} L{self.lap_number}>"
