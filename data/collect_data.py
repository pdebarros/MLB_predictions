"""
Collect Statcast pitch-by-pitch data and build pitcher/batter index maps.
If the CSV already exists, load from file and skip the Statcast fetch.

Run from project root: python -m data.collect_data
Or: python data/collect_data.py
"""
from pathlib import Path

import pandas as pd

# Logs folder inside data/
_data_dir = Path(__file__).resolve().parent
_logs_dir = _data_dir / "logs"
_logs_dir.mkdir(exist_ok=True)
csv_path = _logs_dir / "statcast_pitches.csv"

loaded_from_file = csv_path.exists()
if loaded_from_file:
    print(f"Loading existing data from {csv_path}")
    gbg_data = pd.read_csv(csv_path)
else:
    from pybaseball import statcast

    gbg_data = statcast(start_dt="2025-04-01", end_dt="2025-06-30")

# Get unique MLBAM IDs for each role
# It is important to treat them separately even if some players (Ohtani) do both.
unique_pitchers = gbg_data["pitcher"].unique().tolist()
unique_batters = gbg_data["batter"].unique().tolist()

# Create pitcher maps
pitcher_id_to_idx = {id: i for i, id in enumerate(unique_pitchers)}
idx_to_pitcher_id = {i: id for id, i in pitcher_id_to_idx.items()}

# Create batter maps
batter_id_to_idx = {id: i for i, id in enumerate(unique_batters)}
idx_to_batter_id = {i: id for id, i in batter_id_to_idx.items()}

print(f"Mapped {len(pitcher_id_to_idx)} pitchers and {len(batter_id_to_idx)} batters.")

# Map the IDs to their new indices
gbg_data["pitcher_idx"] = gbg_data["pitcher"].map(pitcher_id_to_idx)
gbg_data["batter_idx"] = gbg_data["batter"].map(batter_id_to_idx)

if not loaded_from_file:
    gbg_data.to_csv(csv_path, index=False)
    print(f"Saved {len(gbg_data)} rows to {csv_path}")

# Verify the first few rows
print(gbg_data[["pitcher", "pitcher_idx", "batter", "batter_idx"]].head())
