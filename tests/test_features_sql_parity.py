"""
SQL / pandas parity tests.

Runs the generated SQL (db/metrics_view.sql's content, regenerated fresh
via build_metrics_view_sql()) against SQLite — with a regr_slope window
aggregate and GREATEST/LEAST scalar shims registered to stand in for
PostgreSQL's, since SQLite lacks them natively — and asserts it matches the
pandas implementation in f1_pit_window.features.build_features exactly.
Only the four original metrics are SQL-backed (the extension features added
in this project are pandas-only — see test_build_features.py); those four
are what's checked here.
"""

from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

from f1_pit_window.features.build_features import build_metrics_view_sql


class _RegrSlope:
    """Incremental OLS slope of y on x, implementing SQLite's window-aggregate protocol."""

    def __init__(self) -> None:
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_xy = 0.0
        self.sum_xx = 0.0

    def step(self, y, x) -> None:
        if y is None or x is None:
            return
        self.n += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_xy += x * y
        self.sum_xx += x * x

    def inverse(self, y, x) -> None:
        if y is None or x is None:
            return
        self.n -= 1
        self.sum_x -= x
        self.sum_y -= y
        self.sum_xy -= x * y
        self.sum_xx -= x * x

    def value(self):
        if self.n < 2:
            return None
        denom = self.n * self.sum_xx - self.sum_x**2
        if denom == 0:
            return None
        return (self.n * self.sum_xy - self.sum_x * self.sum_y) / denom

    def finalize(self):
        return self.value()


@pytest.fixture
def sqlite_conn(laps_with_features: pd.DataFrame) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.create_window_function("regr_slope", 2, _RegrSlope)
    conn.create_function("GREATEST", 2, lambda a, b: None if a is None or b is None else max(a, b))
    conn.create_function("LEAST", 2, lambda a, b: None if a is None or b is None else min(a, b))

    load_columns = [
        "session_key", "driver_number", "lap_number", "stint_number",
        "lap_time_seconds", "is_pit_lap", "pit_in_time", "pit_out_time",
    ]
    laps_with_features[load_columns].to_sql("laps", conn, index=False)
    yield conn
    conn.close()


def _run_generated_view(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = build_metrics_view_sql(view_name="canonical_lap_metrics", source_table="laps")
    sqlite_sql = sql.replace("CREATE OR REPLACE VIEW", "CREATE VIEW")
    conn.executescript(sqlite_sql)
    return pd.read_sql("SELECT * FROM canonical_lap_metrics", conn)


class TestSqlPandasParity:
    def test_view_builds_and_runs_on_sqlite(self, sqlite_conn):
        assert len(_run_generated_view(sqlite_conn)) == 40

    @pytest.mark.parametrize("metric_key", ["gap_to_leader", "tyre_age", "pit_delta", "degradation_rate"])
    def test_metric_matches_pandas_implementation(self, sqlite_conn, laps_with_features, metric_key):
        sort_cols = ["session_key", "driver_number", "lap_number"]
        sql_result = _run_generated_view(sqlite_conn).sort_values(sort_cols).reset_index(drop=True)
        pandas_result = laps_with_features.sort_values(sort_cols).reset_index(drop=True)
        pd.testing.assert_series_equal(
            sql_result[metric_key].astype(float), pandas_result[metric_key].astype(float),
            check_names=False, atol=1e-6,
        )

    def test_view_sql_is_deterministic_text_for_the_same_registry_state(self):
        assert build_metrics_view_sql() == build_metrics_view_sql()


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"),
    reason="Live PostgreSQL parity check — set DATABASE_URL to a real Postgres instance to run it.",
)
def test_parity_against_live_postgres(laps_with_features):
    from sqlalchemy import create_engine

    # Isolated table/view names: this test must never touch the real "laps"
    # table or "canonical_lap_metrics" view on whatever DATABASE_URL points
    # to (e.g. a developer's already-ingested dev database) — it previously
    # used to_sql(..., if_exists="replace") directly against "laps", which
    # silently dropped and rebuilt that table with only these 8 columns and
    # left the fixture rows behind. See docs/decisions/ for the writeup.
    test_table = "laps_sql_parity_test"
    test_view = "canonical_lap_metrics_sql_parity_test"

    engine = create_engine(os.environ["DATABASE_URL"])
    load_columns = [
        "session_key", "driver_number", "lap_number", "stint_number",
        "lap_time_seconds", "is_pit_lap", "pit_in_time", "pit_out_time",
    ]
    try:
        laps_with_features[load_columns].to_sql(test_table, engine, if_exists="replace", index=False)

        with engine.begin() as conn:
            conn.exec_driver_sql(build_metrics_view_sql(view_name=test_view, source_table=test_table))
            sql_result = pd.read_sql(f"SELECT * FROM {test_view}", conn)
    finally:
        with engine.begin() as conn:
            conn.exec_driver_sql(f"DROP VIEW IF EXISTS {test_view}")
            conn.exec_driver_sql(f"DROP TABLE IF EXISTS {test_table}")

    sort_cols = ["session_key", "driver_number", "lap_number"]
    for key in ("gap_to_leader", "tyre_age", "pit_delta", "degradation_rate"):
        pd.testing.assert_series_equal(
            sql_result.sort_values(sort_cols)[key].reset_index(drop=True).astype(float),
            laps_with_features.sort_values(sort_cols)[key].reset_index(drop=True).astype(float),
            check_names=False, atol=1e-6,
        )
