"""
Convert pitch-by-pitch Statcast data to plate appearances with outcome mapping.
"""
from typing import Optional

import pandas as pd

# Outcome classes for the model (order matters for label indices)
DEFAULT_OUTCOME_CLASSES = ["strikeout", "walk", "hit", "out"]

# Map raw Statcast events to outcome buckets
EVENT_TO_OUTCOME = {
    # Hits
    "single": "hit",
    "double": "hit",
    "triple": "hit",
    "home_run": "hit",
    # Strikeouts
    "strikeout": "strikeout",
    "strikeout_double_play": "strikeout",
    # Walks
    "walk": "walk",
    "hit_by_pitch": "walk",
    "intent_walk": "walk",
    "catcher_interf": "walk",
    # Outs (all other fielding plays)
    "field_out": "out",
    "force_out": "out",
    "grounded_into_double_play": "out",
    "sac_fly": "out",
    "sac_bunt": "out",
    "field_error": "out",
    "double_play": "out",
    "fielders_choice": "out",
    "fielders_choice_out": "out",
    "sac_fly_double_play": "out",
}


def _map_event_to_outcome(events: pd.Series) -> pd.Series:
    """Map raw events to outcome classes. Unknown/NaN -> None (to drop)."""
    return events.map(lambda x: EVENT_TO_OUTCOME.get(x) if pd.notna(x) and x else None)


def build_plate_appearances(
    pitches_df: pd.DataFrame,
    outcome_classes: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Convert pitch-by-pitch data to one row per plate appearance with mapped outcome.

    Args:
        pitches_df: Raw Statcast pitch data with columns game_pk, inning, at_bat_number,
                    pitcher, batter, events, game_date, stand, p_throws, etc.
        outcome_classes: Target outcome buckets. Defaults to DEFAULT_OUTCOME_CLASSES.

    Returns:
        DataFrame with one row per PA: game_date, game_pk, inning, at_bat_number,
        pitcher, batter, target_class, stand, p_throws, and preserved contextual cols.
    """
    outcome_classes = outcome_classes or DEFAULT_OUTCOME_CLASSES

    # Ensure required columns exist
    required = ["game_pk", "inning", "at_bat_number", "pitcher", "batter", "game_date"]
    missing = [c for c in required if c not in pitches_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by pitch order (last pitch = outcome)
    df = pitches_df.copy()
    if "pitch_number" in df.columns:
        df = df.sort_values(
            ["game_pk", "inning", "at_bat_number", "pitch_number"]
        )
    else:
        df = df.sort_values(["game_pk", "inning", "at_bat_number"])

    # Take last pitch of each PA
    pa = df.groupby(
        ["game_pk", "inning", "at_bat_number"], as_index=False
    ).last()
    pa = pa.copy()  # defragment to avoid PerformanceWarning

    # Map events to outcome classes
    pa["target_class"] = _map_event_to_outcome(pa["events"])

    # Drop PAs with unknown outcome
    pa = pa.dropna(subset=["target_class"])
    pa = pa[pa["target_class"].isin(outcome_classes)]

    # Select and order columns for downstream use
    keep_cols = [
        "game_date",
        "game_pk",
        "inning",
        "at_bat_number",
        "pitcher",
        "batter",
        "target_class",
    ]
    optional_cols = [
        "stand",
        "p_throws",
        "outs_when_up",
        "on_1b",
        "on_2b",
        "on_3b",
        "home_score",
        "away_score",
        "bat_score",
        "fld_score",
        "pitcher_days_since_prev_game",
        "batter_days_since_prev_game",
    ]
    for c in optional_cols:
        if c in pa.columns:
            keep_cols.append(c)

    result = pa[[c for c in keep_cols if c in pa.columns]].copy()
    return result.reset_index(drop=True)
