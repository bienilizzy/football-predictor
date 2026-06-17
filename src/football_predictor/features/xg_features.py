"""Rolling xG-based features: attacking/defensive quality and over/under-performance vs xG."""
from __future__ import annotations

import pandas as pd

from football_predictor.features.form import rolling_pre_match

DEFAULT_WINDOWS: tuple[int, ...] = (5, 10)


def add_xg_features(team_log: pd.DataFrame, windows: tuple[int, ...] = DEFAULT_WINDOWS) -> pd.DataFrame:
    """Add rolling xG-for/xG-against and goals-minus-xG (over/underperformance) columns.

    `team_log`: one row per (match, team), with columns
        team_id, kickoff_utc, goals_for, xg_for, xg_against
    sorted by team_id then kickoff_utc.

    For each window size N, adds:
        xg_for_N, xg_against_N   -- rolling average xG for/against over last N matches
        xg_overperf_N            -- rolling average (goals_for - xg_for) over last N matches
                                     (positive = team has been scoring more than xG suggests)
    """
    df = team_log.copy()
    df["_goal_diff_vs_xg"] = df["goals_for"] - df["xg_for"]

    by_team = df.groupby("team_id", group_keys=False)
    for window in windows:
        df[f"xg_for_{window}"] = by_team["xg_for"].transform(lambda s, w=window: rolling_pre_match(s, w))
        df[f"xg_against_{window}"] = by_team["xg_against"].transform(lambda s, w=window: rolling_pre_match(s, w))
        df[f"xg_overperf_{window}"] = by_team["_goal_diff_vs_xg"].transform(
            lambda s, w=window: rolling_pre_match(s, w)
        )

    return df.drop(columns=["_goal_diff_vs_xg"])
