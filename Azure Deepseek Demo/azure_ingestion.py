"""
Generate a large synthetic lap-level CSV for pipeline experiments.
Output path is next to this script unless overridden.
"""
from pathlib import Path
import pandas as pd
import random
from datetime import timedelta, datetime

OUTPUT_CSV = Path(__file__).resolve().parent / "race_data.csv"

# Generate synthetic race data
num_rows = 1000000
random.seed(42)
teams = ["Mercedes", "Red Bull", "Ferrari", "McLaren", "Aston Martin"]
tracks = ["Monza", "Silverstone", "Suzuka", "Spa", "Bahrain"]
tires = ["Soft", "Medium", "Hard"]
race_data = []
for _ in range(num_rows):
    race_data.append({
        "race_id": random.randint(1, 50),
        "driver": f"Driver_{random.randint(1, 20)}",
        "team": random.choice(teams),
        "track": random.choice(tracks),
        "lap": random.randint(1, 70),
        "lap_time": round(random.uniform(60, 105), 2),  # seconds
        "tire": random.choice(tires),
        "track_temp": round(random.uniform(20, 50), 1),  # Celsius
        "weather": random.choice(["Sunny", "Rain", "Cloudy"]),
        "pit_stop": random.choice([0, 1])  # 1 if pit stop happened
    })
# Convert to DataFrame and save as CSV
race_df = pd.DataFrame(race_data)
race_df.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote {len(race_df):,} rows to {OUTPUT_CSV}")

