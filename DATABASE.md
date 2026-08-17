## PostgreSQL Database Setup

The `schema.sql` defines the F1 pit strategy database schema for storing feature data and model predictions.

### Quick Start (Docker Compose)

```bash
docker-compose up -d
```

This starts a PostgreSQL 16 container on `localhost:5432` with:
- **Database:** `f1strategy`
- **User:** `f1user`
- **Password:** `f1pass`
- **Auto-schema:** `schema.sql` is applied on first run

### Query Examples

Once the container is running, connect via `psql` or any PostgreSQL client:

```bash
psql -U f1user -d f1strategy -h localhost -p 5432
```

Then run sample queries:

```sql
-- Feature statistics for 2024 races
SELECT
    r.year, r.round, r.name,
    COUNT(*) as num_laps,
    AVG(f.degradation_rate) as avg_deg_rate,
    AVG(f.stint_age_squared) as avg_stint_age2,
    SUM(CASE WHEN f.pit_next_5_laps = 1 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as pit_rate
FROM features f
JOIN races r ON r.race_id = f.race_id
WHERE r.year = 2024
GROUP BY r.year, r.round, r.name
ORDER BY r.round;

-- Top pit predictions for driver 1 in race 5/2024
SELECT
    f.lap_number,
    f.degradation_rate,
    f.stint_age_squared,
    f.race_progress,
    f.lap_time_delta,
    p.pit_probability,
    p.predicted_pit
FROM features f
LEFT JOIN predictions p ON p.feature_id = f.feature_id
JOIN races r ON r.race_id = f.race_id
JOIN drivers d ON d.driver_id = f.driver_id
WHERE r.year = 2024 AND r.round = 5 AND d.driver_number = 1
ORDER BY f.lap_number;

-- Model performance at different thresholds
SELECT
    p.threshold,
    COUNT(*) as total_predictions,
    SUM(p.predicted_pit) as pit_calls,
    ROUND(
        SUM(CASE WHEN p.predicted_pit = 1 AND f.pit_next_5_laps = 1 THEN 1 ELSE 0 END)::NUMERIC /
        NULLIF(SUM(p.predicted_pit), 0), 3
    ) as precision,
    ROUND(
        SUM(CASE WHEN p.predicted_pit = 1 AND f.pit_next_5_laps = 1 THEN 1 ELSE 0 END)::NUMERIC /
        NULLIF(SUM(f.pit_next_5_laps), 0), 3
    ) as recall
FROM predictions p
JOIN features f ON f.feature_id = p.feature_id
GROUP BY p.threshold
ORDER BY p.threshold;
```

### Schema Overview

**tables:**
- `races`: (year, round, name, circuit)
- `drivers`: (driver_number, abbreviation, full_name)
- `features`: 4 model features + pit target per lap
- `predictions`: model predictions at different thresholds

**indexes:** race, driver, lap_number lookups are optimized for performance

### Teardown

```bash
docker-compose down
# Optional: remove volume
docker volume rm f1_pit_strategy_optimization_postgres_data
```

---

For data loading and model training, see `pipeline.py --real-data` to populate features table.
