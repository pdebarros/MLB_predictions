"""
Rolling temporal features for pitchers and batters, plus matchup history for GRU.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureDims:
    """Dimensions of node and edge features."""

    node_dim: int
    edge_dim: int


class RollingTemporalFeatureBuilder:
    """
    Build rolling stats for pitchers and batters from plate appearance data.
    Used for node features in the GAT.
    """

    def __init__(
        self,
        pa_df: pd.DataFrame,
        rolling_window: int = 14,
        outcome_classes: Optional[list[str]] = None,
    ):
        """
        Args:
            pa_df: Plate appearance DataFrame from build_plate_appearances.
                   Must have: game_date, pitcher, batter, target_class.
            rolling_window: Number of prior plate appearances to look back (default: 14).
            outcome_classes: Outcome classes (default: strikeout, walk, hit, out).
        """
        self.pa_df = pa_df.copy()
        self.pa_df["game_date"] = pd.to_datetime(self.pa_df["game_date"])
        self.rolling_window = rolling_window
        self.outcome_classes = outcome_classes or [
            "strikeout",
            "walk",
            "hit",
            "out",
        ]
        self._pitcher_stats: Optional[pd.DataFrame] = None
        self._batter_stats: Optional[pd.DataFrame] = None
        self._matchup_sequences: Optional[dict] = None
        self._build_stats()

    def _build_stats(self) -> None:
        """Precompute rolling stats for all pitcher-date and batter-date pairs."""
        df = self.pa_df.copy()
        df = df.sort_values(
            ["game_date", "game_pk", "inning", "at_bat_number"],
            na_position="last",
        )

        # One-hot outcomes for aggregation
        for oc in self.outcome_classes:
            df[f"is_{oc}"] = (df["target_class"] == oc).astype(float)

        self._pitcher_rolling = self._rolling_aggregate_by_appearances(
            df, "pitcher", ["pa", "strikeout", "walk", "hit", "out"]
        )
        self._batter_rolling = self._rolling_aggregate_by_appearances(
            df, "batter", ["pa", "strikeout", "walk", "hit", "out"]
        )

    def _rolling_aggregate_by_appearances(
        self,
        df: pd.DataFrame,
        player_col: str,
        stat_cols: list[str],
    ) -> pd.DataFrame:
        """
        For each (player, date), compute stats from prior N plate appearances (excluding current day).
        """
        sort_cols = ["game_date", "game_pk", "inning", "at_bat_number"]
        available = [c for c in sort_cols if c in df.columns]
        df = df.sort_values(available, na_position="last")

        result_rows = []
        unique_pairs = df[[player_col, "game_date"]].drop_duplicates()

        for _, pair in unique_pairs.iterrows():
            player_id = pair[player_col]
            d = pair["game_date"]

            # All PAs for this player before this date
            past = df[(df[player_col] == player_id) & (df["game_date"] < d)]
            if past.empty:
                feats = {c: 0.0 for c in stat_cols}
            else:
                # Take last N appearances
                tail = past.tail(self.rolling_window)
                feats = {
                    "pa": len(tail),
                    "strikeout": tail[f"is_{self.outcome_classes[0]}"].sum(),
                    "walk": tail[f"is_{self.outcome_classes[1]}"].sum(),
                    "hit": tail[f"is_{self.outcome_classes[2]}"].sum(),
                    "out": tail[f"is_{self.outcome_classes[3]}"].sum(),
                }

            result_rows.append(
                {
                    player_col: player_id,
                    "game_date": d,
                    **{f"{c}_rolling": feats[c] for c in stat_cols},
                }
            )

        return pd.DataFrame(result_rows)

    def get_pitcher_features(self, pitcher_id: int, game_date: pd.Timestamp) -> np.ndarray:
        """
        Get rolling stat vector for a pitcher on a given date.
        Returns: [pa, strikeout, walk, hit, out] (can add rates if desired).
        """
        row = self._pitcher_rolling[
            (self._pitcher_rolling["pitcher"] == pitcher_id)
            & (self._pitcher_rolling["game_date"] == game_date)
        ]
        if row.empty:
            return np.zeros(5, dtype=np.float32)
        r = row.iloc[0]
        pa = r["pa_rolling"]
        if pa == 0:
            return np.array([0, 0, 0, 0, 0], dtype=np.float32)
        return np.array(
            [
                pa,
                r["strikeout_rolling"] / pa,
                r["walk_rolling"] / pa,
                r["hit_rolling"] / pa,
                r["out_rolling"] / pa,
            ],
            dtype=np.float32,
        )

    def get_batter_features(self, batter_id: int, game_date: pd.Timestamp) -> np.ndarray:
        """Get rolling stat vector for a batter on a given date."""
        row = self._batter_rolling[
            (self._batter_rolling["batter"] == batter_id)
            & (self._batter_rolling["game_date"] == game_date)
        ]
        if row.empty:
            return np.zeros(5, dtype=np.float32)
        r = row.iloc[0]
        pa = r["pa_rolling"]
        if pa == 0:
            return np.array([0, 0, 0, 0, 0], dtype=np.float32)
        return np.array(
            [
                pa,
                r["strikeout_rolling"] / pa,
                r["walk_rolling"] / pa,
                r["hit_rolling"] / pa,
                r["out_rolling"] / pa,
            ],
            dtype=np.float32,
        )

    def get_feature_dims(self) -> FeatureDims:
        """Return dimensions for node and edge features."""
        return FeatureDims(node_dim=5, edge_dim=0)

    def build_matchup_sequences(self) -> dict[tuple[int, int], list[dict]]:
        """
        Build (pitcher, batter) -> list of matchup stat snapshots over time.
        Each snapshot: {pa, avg, k_rate, bb_rate} after each prior meeting.
        Used as GRU input sequence.
        """
        df = self.pa_df.sort_values("game_date")
        sequences: dict[tuple[int, int], list[dict]] = {}

        for (pitcher_id, batter_id), grp in df.groupby(["pitcher", "batter"]):
            grp = grp.sort_values("game_date").reset_index(drop=True)
            snapshots = []
            cum_pa = 0
            cum_hit = 0
            cum_k = 0
            cum_bb = 0

            for _, row in grp.iterrows():
                cum_pa += 1
                cum_hit += 1 if row["target_class"] == "hit" else 0
                cum_k += 1 if row["target_class"] == "strikeout" else 0
                cum_bb += 1 if row["target_class"] == "walk" else 0
                ab = cum_pa - cum_bb  # approximate AB
                snapshots.append({
                    "pa": cum_pa,
                    "avg": cum_hit / ab if ab > 0 else 0.0,
                    "k_rate": cum_k / cum_pa,
                    "bb_rate": cum_bb / cum_pa,
                })
            sequences[(int(pitcher_id), int(batter_id))] = snapshots

        self._matchup_sequences = sequences
        return sequences

    def get_matchup_sequence(
        self, pitcher_id: int, batter_id: int, before_date: pd.Timestamp
    ) -> list[dict]:
        """
        Get matchup stat sequence for (pitcher, batter) prior to a given date.
        Each element is [pa, avg, k_rate, bb_rate] for GRU input.
        """
        if self._matchup_sequences is None:
            self.build_matchup_sequences()

        key = (int(pitcher_id), int(batter_id))
        if key not in self._matchup_sequences:
            return []

        df = self.pa_df
        pair_pas = df[
            (df["pitcher"] == pitcher_id) & (df["batter"] == batter_id)
        ].sort_values("game_date")
        pair_pas = pair_pas[pair_pas["game_date"] < before_date]

        snapshots = []
        cum_pa = cum_hit = cum_k = cum_bb = 0
        for _, row in pair_pas.iterrows():
            cum_pa += 1
            cum_hit += 1 if row["target_class"] == "hit" else 0
            cum_k += 1 if row["target_class"] == "strikeout" else 0
            cum_bb += 1 if row["target_class"] == "walk" else 0
            ab = cum_pa - cum_bb
            snapshots.append(
                [cum_pa, cum_hit / ab if ab > 0 else 0.0, cum_k / cum_pa, cum_bb / cum_pa]
            )
        return snapshots
