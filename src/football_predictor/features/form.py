"""Rolling form features: points-per-game and goals for/against, overall and venue-specific.

All rolling stats use only matches *strictly before* the current one (via
``shift(1)``) so no feature can leak the outcome of the match it describes.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_WINDOWS: tuple[int, ...] = (5, 10)


def rolling_pre_match(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean over the previous `window` values, excluding the current row."""
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def add_form_features(team_log: pd.DataFrame, windows: tuple[int, ...] = DEFAULT_WINDOWS) -> pd.DataFrame:
    """Add rolling points/goals-for/goals-against columns to a team-match log.

    `team_log`: one row per (match, team), with columns
        team_id, kickoff_utc, is_home, goals_for, goals_against, points
    sorted by team_id then kickoff_utc.

    For each window size N, adds:
        form_pts_N, form_gf_N, form_ga_N             -- over the team's last N matches
        venue_form_pts_N, venue_form_gf_N, venue_form_ga_N
                                                       -- over the team's last N matches
                                                          at the same venue (home/away)
    """
    df = team_log.sort_values(["team_id", "kickoff_utc"]).reset_index(drop=True)

    by_team = df.groupby("team_id", group_keys=False)
    by_team_venue = df.groupby(["team_id", "is_home"], group_keys=False)

    for window in windows:
        df[f"form_pts_{window}"] = by_team["points"].transform(lambda s, w=window: rolling_pre_match(s, w))
        df[f"form_gf_{window}"] = by_team["goals_for"].transform(lambda s, w=window: rolling_pre_match(s, w))
        df[f"form_ga_{window}"] = by_team["goals_against"].transform(lambda s, w=window: rolling_pre_match(s, w))

        df[f"venue_form_pts_{window}"] = by_team_venue["points"].transform(
            lambda s, w=window: rolling_pre_match(s, w)
        )
        df[f"venue_form_gf_{window}"] = by_team_venue["goals_for"].transform(
            lambda s, w=window: rolling_pre_match(s, w)
        )
        df[f"venue_form_ga_{window}"] = by_team_venue["goals_against"].transform(
            lambda s, w=window: rolling_pre_match(s, w)
        )

    return df
