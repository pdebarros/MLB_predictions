"""
Build context-only temporal graph data + supervision CSV.

Outputs:
  - data/logs/graph_dataset_context_only.pkl  (graphs with only stats, no labels)
  - data/logs/temporal_training_samples.csv   (all PA samples + outcomes)

This script is standalone and does not alter existing pipelines.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

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
    _add_pair_temporal_stats,
    _add_player_season_stats,
    build_rich_plate_appearances,
)


REL = ("pitcher", "faces", "batter")


def load_or_fetch_pitches() -> pd.DataFrame:
    pitches_path = _logs_dir / "statcast_pitches.csv"
    if pitches_path.exists():
        print(f"Loading existing data from {pitches_path}")
        return pd.read_csv(pitches_path)

    print("Fetching Statcast data...")
    from pybaseball import statcast

    pitches = statcast(start_dt="2025-04-01", end_dt="2025-06-30")
    pitches.to_csv(pitches_path, index=False)
    print(f"Saved {len(pitches):,} rows to {pitches_path}")
    return pitches


def main() -> None:
    print("Loading/fetching pitch data...")
    pitches = load_or_fetch_pitches()
    print(f"  {len(pitches):,} pitch rows")

    print("\nBuilding rich PA table...")
    pa = build_rich_plate_appearances(pitches, outcome_classes=DEFAULT_OUTCOME_CLASSES)
    pa["game_date"] = pd.to_datetime(pa["game_date"])
    pa = pa.sort_values(["game_date", "game_pk", "inning", "at_bat_number"]).reset_index(drop=True)
    print(f"  {len(pa):,} PAs")

    print("Computing temporal pair/player stats...")
    pa = _add_pair_temporal_stats(pa)
    pa = _add_player_season_stats(pa)

    classes = [c for c in DEFAULT_OUTCOME_CLASSES if c in pa["target_class"].unique().tolist()]
    label_to_idx = {c: i for i, c in enumerate(classes)}
    pa["target_idx"] = pa["target_class"].map(label_to_idx).astype(np.int64)

    feature_builder = RollingTemporalFeatureBuilder(
        pa[
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

    print("Building per-day context-only graphs + supervision rows...")
    dates = sorted(pa["game_date"].unique().tolist())
    graphs: list[HeteroData] = []
    rows_out: list[dict] = []

    for t, d in enumerate(dates):
        day = pa[pa["game_date"] == d].copy()
        day = day.sort_values(["game_pk", "inning", "at_bat_number"]).reset_index(drop=True)

        pitchers = day["pitcher"].unique()
        batters = day["batter"].unique()
        p2i = {int(p): i for i, p in enumerate(pitchers)}
        b2i = {int(b): i for i, b in enumerate(batters)}

        pitcher_x = np.stack(
            [feature_builder.get_pitcher_features(int(p), d) for p in pitchers],
            axis=0,
        ).astype(np.float32)
        batter_x = np.stack(
            [feature_builder.get_batter_features(int(b), d) for b in batters],
            axis=0,
        ).astype(np.float32)

        pitcher_x_season = []
        for p in pitchers:
            r = day[day["pitcher"] == p].iloc[0]
            pitcher_x_season.append(
                [
                    r["pit_season_pa"],
                    r["pit_season_k_rate"],
                    r["pit_season_bb_rate"],
                    r["pit_season_era"],
                    r["pit_season_fip"],
                ]
            )
        batter_x_season = []
        for b in batters:
            r = day[day["batter"] == b].iloc[0]
            batter_x_season.append(
                [
                    r["bat_season_pa"],
                    r["bat_season_ba"],
                    r["bat_season_ops"],
                    r["bat_season_k_rate"],
                    r["bat_season_babip"],
                    r["bat_season_bb_rate"],
                ]
            )

        edge_src = []
        edge_dst = []
        edge_attr = []
        for edge_local_idx, (_, r) in enumerate(day.iterrows()):
            pi = p2i[int(r["pitcher"])]
            bi = b2i[int(r["batter"])]
            edge_src.append(pi)
            edge_dst.append(bi)
            edge_attr.append(
                [
                    r["h2p_pa"],
                    r["h2p_ba"],
                    r["h2p_ops"],
                    r["h2p_k_rate"],
                    r["h2p_babip"],
                    r["h2p_bb_rate"],
                    r["p2h_pa"],
                    r["p2h_k_rate"],
                    r["p2h_bb_rate"],
                    r["p2h_era"],
                    r["p2h_fip"],
                ]
            )

            rows_out.append(
                {
                    "time_idx": t,
                    "game_date": str(pd.Timestamp(d).date()),
                    "game_pk": int(r["game_pk"]),
                    "inning": int(r["inning"]),
                    "at_bat_number": int(r["at_bat_number"]),
                    "pitcher": int(r["pitcher"]),
                    "batter": int(r["batter"]),
                    "target_class": r["target_class"],
                    "target_idx": int(r["target_idx"]),
                    "pitcher_local_idx": int(pi),
                    "batter_local_idx": int(bi),
                    "edge_local_idx": int(edge_local_idx),
                }
            )

        data = HeteroData()
        data["pitcher"].x = torch.from_numpy(pitcher_x).float()
        data["batter"].x = torch.from_numpy(batter_x).float()
        data["pitcher"].x_season = torch.tensor(np.asarray(pitcher_x_season, dtype=np.float32))
        data["batter"].x_season = torch.tensor(np.asarray(batter_x_season, dtype=np.float32))

        if len(edge_src) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_t = torch.zeros((0, len(HITTER_EDGE_FEATURE_NAMES) + len(PITCHER_EDGE_FEATURE_NAMES)))
        else:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_attr_t = torch.tensor(np.asarray(edge_attr, dtype=np.float32), dtype=torch.float32)
        data[REL].edge_index = edge_index
        data[REL].edge_attr = edge_attr_t
        data["pitcher"].num_nodes = len(pitchers)
        data["batter"].num_nodes = len(batters)
        graphs.append(data)

    samples_df = pd.DataFrame(rows_out)
    samples_csv_path = _logs_dir / "temporal_training_samples.csv"
    samples_df.to_csv(samples_csv_path, index=False)

    payload = {
        "graphs": graphs,
        "graph_dates": [str(pd.Timestamp(d).date()) for d in dates],
        "edge_feature_names": HITTER_EDGE_FEATURE_NAMES + PITCHER_EDGE_FEATURE_NAMES,
        "pitcher_season_feature_names": PITCHER_SEASON_FEATURE_NAMES,
        "batter_season_feature_names": BATTER_SEASON_FEATURE_NAMES,
        "label_to_idx": label_to_idx,
        "has_edge_labels": False,
    }
    out_pkl_path = _logs_dir / "graph_dataset_context_only.pkl"
    with open(out_pkl_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nSaved {len(graphs)} context-only graphs to {out_pkl_path}")
    print(f"Saved {len(samples_df):,} supervision rows to {samples_csv_path}")
    print(f"Classes: {label_to_idx}")


if __name__ == "__main__":
    main()

