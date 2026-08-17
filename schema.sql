-- F1 Pit Strategy Database Schema
-- PostgreSQL 16+ with indexes for pit timing model

-- Races dimension
CREATE TABLE IF NOT EXISTS races (
    race_id      SERIAL PRIMARY KEY,
    year         SMALLINT     NOT NULL,
    round        SMALLINT     NOT NULL,
    name         VARCHAR(80),
    circuit      VARCHAR(80),
    UNIQUE (year, round)
);

-- Drivers dimension
CREATE TABLE IF NOT EXISTS drivers (
    driver_id     SERIAL PRIMARY KEY,
    driver_number SMALLINT     NOT NULL UNIQUE,
    abbreviation  CHAR(3),
    full_name     VARCHAR(80)
);

-- Engineered features (one row per lap, 4 model features)
CREATE TABLE IF NOT EXISTS features (
    feature_id          BIGSERIAL PRIMARY KEY,
    race_id             INT          REFERENCES races(race_id) ON DELETE CASCADE,
    driver_id           INT          REFERENCES drivers(driver_id) ON DELETE CASCADE,
    lap_number          SMALLINT     NOT NULL,
    -- 4 model features --
    degradation_rate    REAL         NOT NULL,   -- s/lap (OLS slope per stint)
    stint_age_squared   REAL         NOT NULL,   -- TyreLife^2
    race_progress       REAL         NOT NULL,   -- LapNumber / TotalLaps  [0..1]
    lap_time_delta      REAL         NOT NULL,   -- LapTime - DriverMedian  (s)
    -- target --
    pit_next_5_laps     SMALLINT     NOT NULL    -- 0 or 1
);

-- Model predictions
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id       BIGSERIAL PRIMARY KEY,
    feature_id          BIGINT       REFERENCES features(feature_id) ON DELETE CASCADE,
    model_version       VARCHAR(20)  NOT NULL DEFAULT 'xgb-v1',
    threshold           REAL         NOT NULL,
    pit_probability     REAL         NOT NULL,
    predicted_pit       SMALLINT     NOT NULL,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_features_race   ON features (race_id);
CREATE INDEX IF NOT EXISTS idx_features_driver ON features (driver_id);
CREATE INDEX IF NOT EXISTS idx_features_lap    ON features (race_id, driver_id, lap_number);
CREATE INDEX IF NOT EXISTS idx_predictions_feature ON predictions (feature_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions (created_at DESC);

-- Grant read access for analysis
GRANT SELECT ON ALL TABLES IN SCHEMA public TO f1user;
