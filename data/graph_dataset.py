"""
Bipartite HeteroData dataset for pitcher-batter matchup prediction.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from torch_geometric.data import HeteroData

from .features import FeatureDims, RollingTemporalFeatureBuilder


@dataclass
class GraphDatasetConfig:
    """Configuration for BipartitePAHeteroDataset."""

    rolling_window: int = 14
    neighbor_top_k: int = 10  # reserved for future use


class BipartitePAHeteroDataset(Dataset):
    """
    Dataset yielding HeteroData bipartite graphs (pitcher -> batter) for each day.

    Each sample is one day's worth of plate appearances as a graph.
    """

    def __init__(
        self,
        pa_df: pd.DataFrame,
        feature_builder: RollingTemporalFeatureBuilder,
        label_to_idx: dict[str, int],
        config: Optional[GraphDatasetConfig] = None,
    ):
        """
        Args:
            pa_df: Plate appearance DataFrame (train or test slice).
            feature_builder: Provides pitcher/batter node features.
            label_to_idx: Maps outcome class str -> int (e.g. {"strikeout": 0, "walk": 1, ...}).
            config: Dataset config. Uses defaults if None.
        """
        self.pa_df = pa_df.copy()
        self.pa_df["game_date"] = pd.to_datetime(self.pa_df["game_date"])
        self.feature_builder = feature_builder
        self.label_to_idx = label_to_idx
        self.config = config or GraphDatasetConfig()

        # One HeteroData per unique date
        self._dates = sorted(self.pa_df["game_date"].unique().tolist())

    def __len__(self) -> int:
        return len(self._dates)

    def __getitem__(self, idx: int) -> HeteroData:
        game_date = self._dates[idx]
        day_pa = self.pa_df[self.pa_df["game_date"] == game_date]

        pitchers = day_pa["pitcher"].unique()
        batters = day_pa["batter"].unique()
        pitcher_to_idx = {p: i for i, p in enumerate(pitchers)}
        batter_to_idx = {b: i for i, b in enumerate(batters)}

        node_dim = self.feature_builder.get_feature_dims().node_dim

        # Node features
        pitcher_x = np.stack(
            [
                self.feature_builder.get_pitcher_features(int(p), game_date)
                for p in pitchers
            ],
            axis=0,
        )
        batter_x = np.stack(
            [
                self.feature_builder.get_batter_features(int(b), game_date)
                for b in batters
            ],
            axis=0,
        )

        # Edges: one per PA, (pitcher_idx, batter_idx)
        edge_src = []
        edge_dst = []
        edge_y = []
        for _, row in day_pa.iterrows():
            pi = pitcher_to_idx[row["pitcher"]]
            bi = batter_to_idx[row["batter"]]
            label = self.label_to_idx.get(row["target_class"])
            if label is None:
                continue
            edge_src.append(pi)
            edge_dst.append(bi)
            edge_y.append(label)

        if len(edge_src) == 0:
            # No valid edges (e.g. unknown target_class)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            y = torch.zeros(0, dtype=torch.long)
        else:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            y = torch.tensor(edge_y, dtype=torch.long)

        data = HeteroData()
        data["pitcher"].x = torch.from_numpy(pitcher_x).float()
        data["batter"].x = torch.from_numpy(batter_x).float()
        data["pitcher", "faces", "batter"].edge_index = edge_index
        data["pitcher", "faces", "batter"].y = y

        # Store batch / num_nodes for batching
        data["pitcher"].num_nodes = len(pitchers)
        data["batter"].num_nodes = len(batters)

        return data
