"""
Shared fixtures.

The synthetic race in ``synthetic_laps`` looks like real FastF1 output —
raw column names, mixed compound spellings across two "seasons",
timedelta-typed lap times, two pit stops, and planted defects (a dropped
sensor run, an outlier lap, a duplicate lap number) — so tests exercise the
pipeline the way real ingest would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_pit_window.data.cleaning import normalize_laps
from f1_pit_window.features.build_features import compute_all

RNG = np.random.RandomState(20260812)


def _make_stint(driver_number: int, stint_number: int, start_lap: int, n_laps: int, base_pace: float, deg_rate: float):
    rows = []
    for i in range(n_laps):
        lap_number = start_lap + i
        lap_time = base_pace + deg_rate * i + RNG.normal(0, 0.15)
        rows.append({
            "DriverNumber": str(driver_number),
            "Driver": f"D{driver_number:02d}",
            "Team": "Team A" if driver_number % 2 == 0 else "Team B",
            "LapNumber": lap_number,
            "LapTime": pd.to_timedelta(lap_time, unit="s"),
            "Sector1Time": pd.to_timedelta(lap_time / 3, unit="s"),
            "Sector2Time": pd.to_timedelta(lap_time / 3, unit="s"),
            "Sector3Time": pd.to_timedelta(lap_time / 3, unit="s"),
            "Compound": "HYPERSOFT" if driver_number == 1 else "SOFT",
            "TyreLife": i,
            "Stint": stint_number,
            "Position": driver_number,
            "PitInTime": pd.NaT,
            "PitOutTime": pd.NaT,
            "AirTemp": 28.0,
            "TrackTemp": 41.0,
            "WindSpeed": 5.0,
            "Humidity": 55.0,
            "Rainfall": False,
        })
    return rows


@pytest.fixture
def synthetic_laps() -> pd.DataFrame:
    """
    Two drivers, one pit stop each, one 20-lap session, FastF1-shaped columns.

    Driver 1: pits after lap 10. Driver 2: pits after lap 12. Planted
    defects: lap 5 (driver 1) is a ~2 min outlier; TrackTemp missing for
    driver 2 laps 8-9; a duplicated lap 15 row for driver 1 is added by the
    ``duplicate_lap_laps`` fixture, not here.
    """
    rows = []
    rows += _make_stint(1, 1, 1, 10, base_pace=92.0, deg_rate=0.08)
    rows += _make_stint(1, 2, 11, 10, base_pace=91.5, deg_rate=0.10)
    rows += _make_stint(2, 1, 1, 12, base_pace=93.0, deg_rate=0.05)
    rows += _make_stint(2, 2, 13, 8, base_pace=92.5, deg_rate=0.07)

    df = pd.DataFrame(rows)
    df["Season"] = 2024
    df["RoundNumber"] = 1

    mask = (df["DriverNumber"] == "1") & (df["LapNumber"] == 5)
    df.loc[mask, "LapTime"] = pd.to_timedelta(125.0, unit="s")

    mask = (df["DriverNumber"] == "2") & (df["LapNumber"].isin([8, 9]))
    df.loc[mask, "TrackTemp"] = np.nan

    df.loc[(df["DriverNumber"] == "1") & (df["LapNumber"] == 10), "PitInTime"] = pd.to_timedelta(600, unit="s")
    df.loc[(df["DriverNumber"] == "1") & (df["LapNumber"] == 11), "PitOutTime"] = pd.to_timedelta(622, unit="s")
    df.loc[(df["DriverNumber"] == "2") & (df["LapNumber"] == 12), "PitInTime"] = pd.to_timedelta(700, unit="s")
    df.loc[(df["DriverNumber"] == "2") & (df["LapNumber"] == 13), "PitOutTime"] = pd.to_timedelta(723.4, unit="s")

    return df.reset_index(drop=True)


@pytest.fixture
def normalized_laps(synthetic_laps: pd.DataFrame) -> pd.DataFrame:
    return normalize_laps(synthetic_laps, session_key="2024-01")


@pytest.fixture
def laps_with_features(normalized_laps: pd.DataFrame) -> pd.DataFrame:
    """``normalized_laps`` with every current canonical feature attached."""
    features = compute_all(normalized_laps)
    return pd.concat([normalized_laps.reset_index(drop=True), features.reset_index(drop=True)], axis=1)


@pytest.fixture
def duplicate_lap_laps(normalized_laps: pd.DataFrame) -> pd.DataFrame:
    dup = normalized_laps[(normalized_laps["driver_number"] == 1) & (normalized_laps["lap_number"] == 6)].copy()
    return pd.concat([normalized_laps, dup], ignore_index=True)
