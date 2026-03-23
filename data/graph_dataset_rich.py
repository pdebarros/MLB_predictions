"""
Richer bipartite HeteroData dataset with temporal matchup/season statistics.

This module is intentionally separate from the existing graph pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from .build_plate_appearances import DEFAULT_OUTCOME_CLASSES, EVENT_TO_OUTCOME
from .features import RollingTemporalFeatureBuilder


HITTER_EDGE_FEATURE_NAMES = [
    "h2p_pa",
    "h2p_ba",
    "h2p_ops",
    "h2p_k_rate",
    "h2p_babip",
    "h2p_bb_rate",
]

PITCHER_EDGE_FEATURE_NAMES = [
    "p2h_pa",
    "p2h_k_rate",
    "p2h_bb_rate",
    "p2h_era",
    "p2h_fip",
]

BATTER_SEASON_FEATURE_NAMES = [
    "season_pa",
    "season_ba",
    "season_ops",
    "season_k_rate",
    "season_babip",
    "season_bb_rate",
]

PITCHER_SEASON_FEATURE_NAMES = [
    "season_pa",
    "season_k_rate",
    "season_bb_rate",
    "season_era",
    "season_fip",
]


def _safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    out = pd.Series(np.zeros(len(n), dtype=np.float32), index=n.index)
    mask = d > 0
    out.loc[mask] = (n.loc[mask] / d.loc[mask]).astype(np.float32)
    return out


def _map_outs_recorded(event: str) -> int:
    if event in {"grounded_into_double_play", "double_play", "sac_fly_double_play"}:
        return 2
    if event in {"triple_play"}:
        return 3
    if event in {
        "strikeout",
        "strikeout_double_play",
        "field_out",
        "force_out",
        "sac_fly",
        "sac_bunt",
        "fielders_choice_out",
        "double_play",
        "grounded_into_double_play",
        "sac_fly_double_play",
    }:
        return 1
    return 0


def build_rich_plate_appearances(
    pitches_df: pd.DataFrame,
    outcome_classes: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    One row per PA with extra event-derived columns used for richer temporal stats.
    """
    outcome_classes = outcome_classes or DEFAULT_OUTCOME_CLASSES

    required = ["game_pk", "inning", "at_bat_number", "pitcher", "batter", "game_date"]
    missing = [c for c in required if c not in pitches_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = pitches_df.copy()
    if "pitch_number" in df.columns:
        df = df.sort_values(["game_date", "game_pk", "inning", "at_bat_number", "pitch_number"])
    else:
        df = df.sort_values(["game_date", "game_pk", "inning", "at_bat_number"])

    pa = df.groupby(["game_pk", "inning", "at_bat_number"], as_index=False).last().copy()
    pa["game_date"] = pd.to_datetime(pa["game_date"])

    pa["target_class"] = pa["events"].map(
        lambda x: EVENT_TO_OUTCOME.get(x) if pd.notna(x) and x else None
    )
    pa = pa.dropna(subset=["target_class"])
    pa = pa[pa["target_class"].isin(outcome_classes)].copy()

    event = pa["events"].fillna("").astype(str)
    pa["is_hit"] = event.isin({"single", "double", "triple", "home_run"}).astype(np.int64)
    pa["is_k"] = event.isin({"strikeout", "strikeout_double_play"}).astype(np.int64)
    pa["is_bb"] = event.isin({"walk", "intent_walk", "catcher_interf", "hit_by_pitch"}).astype(
        np.int64
    )
    pa["is_hbp"] = event.isin({"hit_by_pitch"}).astype(np.int64)
    pa["is_hr"] = event.isin({"home_run"}).astype(np.int64)
    pa["is_sf"] = event.isin({"sac_fly", "sac_fly_double_play"}).astype(np.int64)

    # AB proxy: no AB on walks/HBP/sac flies.
    pa["ab"] = (1 - ((pa["is_bb"] + pa["is_hbp"] + pa["is_sf"]) > 0).astype(np.int64)).astype(
        np.int64
    )

    pa["total_bases"] = (
        event.map({"single": 1, "double": 2, "triple": 3, "home_run": 4}).fillna(0).astype(np.int64)
    )

    pa["outs_recorded"] = event.map(_map_outs_recorded).fillna(0).astype(np.int64)

    # Runs allowed on PA for pitcher metrics if score deltas are available.
    if "post_bat_score" in pa.columns and "bat_score" in pa.columns:
        pa["runs_scored"] = (pa["post_bat_score"] - pa["bat_score"]).clip(lower=0).fillna(0).astype(
            np.int64
        )
    elif "post_fld_score" in pa.columns and "fld_score" in pa.columns:
        # Fallback orientation; use whichever delta exists.
        pa["runs_scored"] = (pa["post_fld_score"] - pa["fld_score"]).clip(lower=0).fillna(0).astype(
            np.int64
        )
    else:
        # Conservative fallback when explicit run columns are absent.
        pa["runs_scored"] = pa["is_hr"].astype(np.int64)

    keep_cols = [
        "game_date",
        "game_pk",
        "inning",
        "at_bat_number",
        "pitcher",
        "batter",
        "target_class",
        "events",
        "is_hit",
        "is_k",
        "is_bb",
        "is_hbp",
        "is_hr",
        "is_sf",
        "ab",
        "total_bases",
        "outs_recorded",
        "runs_scored",
    ]
    return pa[keep_cols].sort_values(["game_date", "game_pk", "inning", "at_bat_number"]).reset_index(
        drop=True
    )


def _add_pair_temporal_stats(pa_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-row prior matchup stats for (pitcher, batter) pairs."""
    df = pa_df.copy()
    grp = df.groupby(["pitcher", "batter"], sort=False)

    df["pair_pa_before"] = grp.cumcount().astype(np.int64)
    for c in ["is_hit", "is_k", "is_bb", "is_hbp", "is_hr", "is_sf", "ab", "total_bases", "outs_recorded", "runs_scored"]:
        csum = grp[c].cumsum()
        df[f"pair_{c}_before"] = csum.groupby([df["pitcher"], df["batter"]]).shift(1).fillna(0).astype(
            np.float32
        )

    pa = df["pair_pa_before"].astype(np.float32)
    ab = df["pair_ab_before"]
    hits = df["pair_is_hit_before"]
    k = df["pair_is_k_before"]
    bb = df["pair_is_bb_before"]
    hbp = df["pair_is_hbp_before"]
    hr = df["pair_is_hr_before"]
    sf = df["pair_is_sf_before"]
    tb = df["pair_total_bases_before"]
    outs = df["pair_outs_recorded_before"]
    runs = df["pair_runs_scored_before"]

    ba = _safe_div(hits, ab)
    slg = _safe_div(tb, ab)
    obp_den = ab + bb + hbp + sf
    obp = _safe_div(hits + bb + hbp, obp_den)
    ops = obp + slg
    k_rate = _safe_div(k, pa)
    bb_rate = _safe_div(bb, pa)
    babip = _safe_div(hits - hr, ab - k - hr + sf)

    ip = outs / 3.0
    era = _safe_div(9.0 * runs, ip)
    fip = pd.Series(np.full(len(df), 3.2, dtype=np.float32), index=df.index)
    mask = ip > 0
    fip.loc[mask] = (
        (13.0 * hr.loc[mask] + 3.0 * (bb.loc[mask] + hbp.loc[mask]) - 2.0 * k.loc[mask]) / ip.loc[mask]
        + 3.2
    ).astype(np.float32)

    df["h2p_pa"] = pa
    df["h2p_ba"] = ba
    df["h2p_ops"] = ops.astype(np.float32)
    df["h2p_k_rate"] = k_rate
    df["h2p_babip"] = babip
    df["h2p_bb_rate"] = bb_rate

    df["p2h_pa"] = pa
    df["p2h_k_rate"] = k_rate
    df["p2h_bb_rate"] = bb_rate
    df["p2h_era"] = era.astype(np.float32)
    df["p2h_fip"] = fip.astype(np.float32)

    return df


def _add_player_season_stats(pa_df: pd.DataFrame) -> pd.DataFrame:
    """Add season-to-date stats (prior to this PA) for batters and pitchers."""
    df = pa_df.copy()

    def add_for_player(player_col: str, prefix: str) -> None:
        grp = df.groupby(player_col, sort=False)
        df[f"{prefix}_pa_before"] = grp.cumcount().astype(np.int64)
        for c in ["is_hit", "is_k", "is_bb", "is_hbp", "is_hr", "is_sf", "ab", "total_bases", "outs_recorded", "runs_scored"]:
            csum = grp[c].cumsum()
            df[f"{prefix}_{c}_before"] = csum.groupby(df[player_col]).shift(1).fillna(0).astype(np.float32)

    add_for_player("batter", "bat")
    add_for_player("pitcher", "pit")

    # Batter season metrics
    b_pa = df["bat_pa_before"].astype(np.float32)
    b_ab = df["bat_ab_before"]
    b_h = df["bat_is_hit_before"]
    b_k = df["bat_is_k_before"]
    b_bb = df["bat_is_bb_before"]
    b_hbp = df["bat_is_hbp_before"]
    b_hr = df["bat_is_hr_before"]
    b_sf = df["bat_is_sf_before"]
    b_tb = df["bat_total_bases_before"]

    b_ba = _safe_div(b_h, b_ab)
    b_slg = _safe_div(b_tb, b_ab)
    b_obp = _safe_div(b_h + b_bb + b_hbp, b_ab + b_bb + b_hbp + b_sf)
    b_ops = (b_obp + b_slg).astype(np.float32)
    b_k_rate = _safe_div(b_k, b_pa)
    b_bb_rate = _safe_div(b_bb, b_pa)
    b_babip = _safe_div(b_h - b_hr, b_ab - b_k - b_hr + b_sf)

    df["bat_season_pa"] = b_pa
    df["bat_season_ba"] = b_ba
    df["bat_season_ops"] = b_ops
    df["bat_season_k_rate"] = b_k_rate
    df["bat_season_babip"] = b_babip
    df["bat_season_bb_rate"] = b_bb_rate

    # Pitcher season metrics
    p_pa = df["pit_pa_before"].astype(np.float32)
    p_k = df["pit_is_k_before"]
    p_bb = df["pit_is_bb_before"]
    p_hbp = df["pit_is_hbp_before"]
    p_hr = df["pit_is_hr_before"]
    p_outs = df["pit_outs_recorded_before"]
    p_runs = df["pit_runs_scored_before"]
    p_ip = p_outs / 3.0

    p_k_rate = _safe_div(p_k, p_pa)
    p_bb_rate = _safe_div(p_bb, p_pa)
    p_era = _safe_div(9.0 * p_runs, p_ip)
    p_fip = pd.Series(np.full(len(df), 3.2, dtype=np.float32), index=df.index)
    mask = p_ip > 0
    p_fip.loc[mask] = (
        (13.0 * p_hr.loc[mask] + 3.0 * (p_bb.loc[mask] + p_hbp.loc[mask]) - 2.0 * p_k.loc[mask]) / p_ip.loc[mask]
        + 3.2
    ).astype(np.float32)

    df["pit_season_pa"] = p_pa
    df["pit_season_k_rate"] = p_k_rate
    df["pit_season_bb_rate"] = p_bb_rate
    df["pit_season_era"] = p_era
    df["pit_season_fip"] = p_fip.astype(np.float32)

    return df


@dataclass
class RichGraphDatasetConfig:
    rolling_window: int = 14
    neighbor_top_k: int = 10


class BipartitePAHeteroRichDataset(Dataset):
    """
    HeteroData with:
    - Node rolling features: x (existing 5D)
    - Node season features: x_season
      * batter: [PA, BA, OPS, K_rate, BABIP, BB_rate]
      * pitcher: [PA, K_rate, BB_rate, ERA, FIP]
    - Edge features edge_attr:
      [h2p_PA, h2p_BA, h2p_OPS, h2p_K_rate, h2p_BABIP, h2p_BB_rate,
       p2h_PA, p2h_K_rate, p2h_BB_rate, p2h_ERA, p2h_FIP]
    """

    def __init__(
        self,
        pa_df_rich: pd.DataFrame,
        feature_builder: RollingTemporalFeatureBuilder,
        label_to_idx: dict[str, int],
        config: Optional[RichGraphDatasetConfig] = None,
    ):
        self.pa_df = pa_df_rich.copy()
        self.pa_df["game_date"] = pd.to_datetime(self.pa_df["game_date"])
        self.pa_df = self.pa_df.sort_values(["game_date", "game_pk", "inning", "at_bat_number"]).reset_index(
            drop=True
        )
        self.feature_builder = feature_builder
        self.label_to_idx = label_to_idx
        self.config = config or RichGraphDatasetConfig()

        self.pa_df = _add_pair_temporal_stats(self.pa_df)
        self.pa_df = _add_player_season_stats(self.pa_df)
        self._dates = sorted(self.pa_df["game_date"].unique().tolist())

    def __len__(self) -> int:
        return len(self._dates)

    def __getitem__(self, idx: int) -> HeteroData:
        game_date = self._dates[idx]
        day_pa = self.pa_df[self.pa_df["game_date"] == game_date].copy()
        day_pa = day_pa.sort_values(["game_pk", "inning", "at_bat_number"]).reset_index(drop=True)

        pitchers = day_pa["pitcher"].unique()
        batters = day_pa["batter"].unique()
        pitcher_to_idx = {p: i for i, p in enumerate(pitchers)}
        batter_to_idx = {b: i for i, b in enumerate(batters)}

        pitcher_x = np.stack(
            [self.feature_builder.get_pitcher_features(int(p), game_date) for p in pitchers], axis=0
        )
        batter_x = np.stack(
            [self.feature_builder.get_batter_features(int(b), game_date) for b in batters], axis=0
        )

        # Season stats from the first PA of the day for each player (state at start of day).
        pitcher_x_season = []
        for p in pitchers:
            r = day_pa[day_pa["pitcher"] == p].iloc[0]
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
            r = day_pa[day_pa["batter"] == b].iloc[0]
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
        edge_y = []
        edge_attr = []
        for _, row in day_pa.iterrows():
            label = self.label_to_idx.get(row["target_class"])
            if label is None:
                continue

            pi = pitcher_to_idx[row["pitcher"]]
            bi = batter_to_idx[row["batter"]]
            edge_src.append(pi)
            edge_dst.append(bi)
            edge_y.append(label)

            feats = [
                row["h2p_pa"],
                row["h2p_ba"],
                row["h2p_ops"],
                row["h2p_k_rate"],
                row["h2p_babip"],
                row["h2p_bb_rate"],
                row["p2h_pa"],
                row["p2h_k_rate"],
                row["p2h_bb_rate"],
                row["p2h_era"],
                row["p2h_fip"],
            ]
            edge_attr.append(feats)

        if len(edge_src) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            y = torch.zeros(0, dtype=torch.long)
            edge_attr_t = torch.zeros((0, len(HITTER_EDGE_FEATURE_NAMES) + len(PITCHER_EDGE_FEATURE_NAMES)))
        else:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            y = torch.tensor(edge_y, dtype=torch.long)
            edge_attr_t = torch.tensor(np.asarray(edge_attr, dtype=np.float32), dtype=torch.float32)

        data = HeteroData()
        data["pitcher"].x = torch.from_numpy(pitcher_x).float()
        data["batter"].x = torch.from_numpy(batter_x).float()
        data["pitcher"].x_season = torch.tensor(np.asarray(pitcher_x_season, dtype=np.float32))
        data["batter"].x_season = torch.tensor(np.asarray(batter_x_season, dtype=np.float32))

        data["pitcher", "faces", "batter"].edge_index = edge_index
        data["pitcher", "faces", "batter"].edge_attr = edge_attr_t
        data["pitcher", "faces", "batter"].y = y

        data["pitcher"].num_nodes = len(pitchers)
        data["batter"].num_nodes = len(batters)
        return data

