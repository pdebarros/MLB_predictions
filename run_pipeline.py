"""
Run the full data pipeline: fetch Statcast (if needed) -> plate appearances -> features.

Reads/writes data/logs/statcast_pitches.csv.
Saves graph dataset to data/logs/graph_dataset.pkl.

Run from project root: python run_pipeline.py
"""
import pickle
import sys
from pathlib import Path

# Project root
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_logs_dir = _project_root / "data" / "logs"
_logs_dir.mkdir(exist_ok=True)

import pandas as pd

from data.build_plate_appearances import build_plate_appearances, DEFAULT_OUTCOME_CLASSES
from data.features import RollingTemporalFeatureBuilder
from data.graph_dataset import BipartitePAHeteroDataset, GraphDatasetConfig


def load_or_fetch_pitches() -> pd.DataFrame:
    """Load pitches from logs/ or fetch from Statcast and save."""
    pitches_path = _logs_dir / "statcast_pitches.csv"
    if pitches_path.exists():
        print(f"Loading existing data from {pitches_path}")
        pitches = pd.read_csv(pitches_path)
    else:
        print("Fetching Statcast data...")
        from pybaseball import statcast
        pitches = statcast(start_dt="2025-04-01", end_dt="2025-06-30")
        pitches.to_csv(pitches_path, index=False)
        print(f"Saved {len(pitches):,} rows to {pitches_path}")
    return pitches


def main():
    print("Loading or fetching pitch data...")
    pitches = load_or_fetch_pitches()
    print(f"  {len(pitches):,} pitch rows\n")

    print("Building plate appearances...")
    pa_df = build_plate_appearances(pitches, outcome_classes=DEFAULT_OUTCOME_CLASSES)
    pa_df["game_date"] = pd.to_datetime(pa_df["game_date"])
    print(f"  {len(pa_df):,} plate appearances")
    print(f"  Outcome distribution:\n{pa_df['target_class'].value_counts()}")

    print("Building rolling features (last 14 appearances, may take a minute)...")
    feature_builder = RollingTemporalFeatureBuilder(
        pa_df,
        rolling_window=14,
        outcome_classes=DEFAULT_OUTCOME_CLASSES,
    )
    dims = feature_builder.get_feature_dims()
    print(f"  Node feature dim: {dims.node_dim}, Edge feature dim: {dims.edge_dim}")

    # Quick sanity check
    sample_pitcher = pa_df["pitcher"].iloc[0]
    sample_batter = pa_df["batter"].iloc[0]
    sample_date = pa_df["game_date"].iloc[100]
    pf = feature_builder.get_pitcher_features(sample_pitcher, sample_date)
    bf = feature_builder.get_batter_features(sample_batter, sample_date)
    print(f"  Sample pitcher features: {pf}")
    print(f"  Sample batter features: {bf}")

    # Build HeteroData dataset
    print("\nBuilding HeteroData dataset...")
    classes = [c for c in DEFAULT_OUTCOME_CLASSES if c in pa_df["target_class"].unique().tolist()]
    label_to_idx = {c: i for i, c in enumerate(classes)}
    cfg = GraphDatasetConfig(rolling_window=14, neighbor_top_k=10)
    ds = BipartitePAHeteroDataset(pa_df, feature_builder, label_to_idx, config=cfg)
    sample = ds[0]
    print(f"  Dataset: {len(ds)} graphs (one per day)")
    print(f"  Sample HeteroData: pitcher nodes={sample['pitcher'].x.shape[0]}, "
          f"batter nodes={sample['batter'].x.shape[0]}, edges={sample['pitcher', 'faces', 'batter'].edge_index.shape[1]}")

    # Save graph dataset to pickle
    graphs_path = _logs_dir / "graph_dataset.pkl"
    graphs_list = [ds[i] for i in range(len(ds))]
    with open(graphs_path, "wb") as f:
        pickle.dump({"graphs": graphs_list, "label_to_idx": label_to_idx}, f)
    print(f"  Saved {len(graphs_list)} graphs to {graphs_path}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
