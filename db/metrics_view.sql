-- Generated from f1_pit_window.features.build_features — do not edit by hand.
-- Registry fingerprint: 5389f4696369
CREATE OR REPLACE VIEW canonical_lap_metrics AS
WITH base AS (
    SELECT
        session_key,
        driver_number,
        lap_number,
        stint_number,
        lap_time_seconds,
        is_pit_lap,
        pit_in_time,
        pit_out_time,
        lap_time_seconds - MIN(lap_time_seconds) OVER (PARTITION BY session_key, lap_number)
            AS lap_gap_seconds,
        ROW_NUMBER() OVER (PARTITION BY session_key, driver_number, stint_number
            ORDER BY lap_number) - 1 AS tyre_age_laps
    FROM laps
)
SELECT
    session_key,
    driver_number,
    lap_number,
    stint_number,
    lap_time_seconds,
    -- degradation_rate@v2 (seconds per lap), 03e837567f12,
    GREATEST(-0.5, LEAST(0.5, COALESCE(regr_slope(lap_time_seconds, tyre_age_laps) OVER (PARTITION BY session_key, driver_number, stint_number), 0))) AS degradation_rate,
    -- gap_to_leader@v2 (seconds), e5207e89f507,
    SUM(lap_gap_seconds) OVER (PARTITION BY session_key, driver_number ORDER BY lap_number ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS gap_to_leader,
    -- pit_delta@v2 (seconds), 4653fb9c1e21,
    CASE WHEN pit_out_time IS NOT NULL AND LAG(pit_in_time) OVER (PARTITION BY session_key, driver_number ORDER BY lap_number) IS NOT NULL AND pit_out_time >= LAG(pit_in_time) OVER (PARTITION BY session_key, driver_number ORDER BY lap_number) THEN pit_out_time - LAG(pit_in_time) OVER (PARTITION BY session_key, driver_number ORDER BY lap_number) END AS pit_delta,
    -- tyre_age@v2 (laps), 8768f8e9e1a4,
    tyre_age_laps AS tyre_age
FROM base;