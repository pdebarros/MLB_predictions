"""
Build a richer graph dataset (separate from the default pipeline).

Output:
  data/logs/graph_dataset_rich.pkl
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_logs_dir = _project_root / "data" / "logs"
_logs_dir.mkdir(exist_ok=True)

from data.build_plate_appearances import DEFAULT_OUTCOME_CLASSES
from data.features import RollingTemporalFeatureBuilder
from data.graph_dataset_rich import (
    BATTER_SEASON_FEATURE_NAMES,
    HITTER_EDGE_FEATURE_NAMES,
    PITCHER_EDGE_FEATURE_NAMES,
    PITCHER_SEASON_FEATURE_NAMES,
    BipartitePAHeteroRichDataset,
    RichGraphDatasetConfig,
    build_rich_plate_appearances,
)


def load_or_fetch_pitches() -> pd.DataFrame:
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


def main() -> None:
    print("Loading or fetching pitch data...")
    pitches = load_or_fetch_pitches()
    print(f"  {len(pitches):,} pitch rows")

    print("\nBuilding richer plate-appearance table...")
    pa_df_rich = build_rich_plate_appearances(pitches, outcome_classes=DEFAULT_OUTCOME_CLASSES)
    print(f"  {len(pa_df_rich):,} plate appearances")
    print(f"  Outcome distribution:\n{pa_df_rich['target_class'].value_counts()}")

    print("\nBuilding rolling feature builder...")
    feature_builder = RollingTemporalFeatureBuilder(
        pa_df_rich[
            [
                "game_date",
                "game_pk",
                "inning",
                "at_bat_number",
                "pitcher",
                "batter",
                "target_class",
            ]
        ].copy(),
        rolling_window=14,
        outcome_classes=DEFAULT_OUTCOME_CLASSES,
    )

    classes = [
        c for c in DEFAULT_OUTCOME_CLASSES if c in pa_df_rich["target_class"].unique().tolist()
    ]
    label_to_idx = {c: i for i, c in enumerate(classes)}

    print("\nBuilding rich HeteroData dataset...")
    cfg = RichGraphDatasetConfig(rolling_window=14, neighbor_top_k=10)
    ds = BipartitePAHeteroRichDataset(
        pa_df_rich=pa_df_rich,
        feature_builder=feature_builder,
        label_to_idx=label_to_idx,
        config=cfg,
    )
    sample = ds[0]
    rel = sample["pitcher", "faces", "batter"]
    print(f"  Dataset: {len(ds)} graphs (one per day)")
    print(
        f"  Sample: pitcher nodes={sample['pitcher'].x.shape[0]}, "
        f"batter nodes={sample['batter'].x.shape[0]}, edges={rel.edge_index.shape[1]}"
    )
    print(f"  Edge attr shape: {rel.edge_attr.shape}")
    print(f"  Pitcher season shape: {sample['pitcher'].x_season.shape}")
    print(f"  Batter season shape: {sample['batter'].x_season.shape}")

    out_path = _logs_dir / "graph_dataset_rich.pkl"
    graphs_list = [ds[i] for i in range(len(ds))]
    payload = {
        "graphs": graphs_list,
        "label_to_idx": label_to_idx,
        "edge_feature_names": HITTER_EDGE_FEATURE_NAMES + PITCHER_EDGE_FEATURE_NAMES,
        "pitcher_season_feature_names": PITCHER_SEASON_FEATURE_NAMES,
        "batter_season_feature_names": BATTER_SEASON_FEATURE_NAMES,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved {len(graphs_list)} rich graphs to {out_path}")


if __name__ == "__main__":
    main()

