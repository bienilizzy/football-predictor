"""Head-to-head features: result/goal history between two specific teams.

For each match, summarizes the last `last_n` previous meetings between the two
teams (regardless of which team was at home in those meetings), expressed from
the *current* home team's perspective.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LAST_N = 5

H2H_COLUMNS = [
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_away_win_rate",
    "h2h_avg_gf_home",
    "h2h_avg_gf_away",
    "h2h_matches_count",
]


def compute_h2h_features(matches: pd.DataFrame, last_n: int = DEFAULT_LAST_N) -> pd.DataFrame:
    """`matches` needs: match_id, kickoff_utc, home_team_id, away_team_id, home_score, away_score.

    Returns one row per match_id with the H2H_COLUMNS above.
    """
    df = matches.sort_values("kickoff_utc").reset_index(drop=True)

    # history[pair] = list of {team_a_id, team_b_id, goals_a, goals_b}, oldest first
    history: dict[frozenset, list[dict]] = {}
    rows: list[dict] = []

    for row in df.itertuples(index=False):
        pair = frozenset((row.home_team_id, row.away_team_id))
        past = history.get(pair, [])[-last_n:]

        if past:
            home_wins = draws = away_wins = 0
            gf_home_total = 0.0
            gf_away_total = 0.0
            for meeting in past:
                a, ga, gb = meeting["team_a_id"], meeting["goals_a"], meeting["goals_b"]
                home_goals = ga if a == row.home_team_id else gb
                away_goals = gb if a == row.home_team_id else ga
                gf_home_total += home_goals
                gf_away_total += away_goals
                if home_goals > away_goals:
                    home_wins += 1
                elif home_goals < away_goals:
                    away_wins += 1
                else:
                    draws += 1
            n = len(past)
            rows.append(
                {
                    "match_id": row.match_id,
                    "h2h_home_win_rate": home_wins / n,
                    "h2h_draw_rate": draws / n,
                    "h2h_away_win_rate": away_wins / n,
                    "h2h_avg_gf_home": gf_home_total / n,
                    "h2h_avg_gf_away": gf_away_total / n,
                    "h2h_matches_count": n,
                }
            )
        else:
            rows.append(
                {
                    "match_id": row.match_id,
                    "h2h_home_win_rate": np.nan,
                    "h2h_draw_rate": np.nan,
                    "h2h_away_win_rate": np.nan,
                    "h2h_avg_gf_home": np.nan,
                    "h2h_avg_gf_away": np.nan,
                    "h2h_matches_count": 0,
                }
            )

        if pd.notna(row.home_score) and pd.notna(row.away_score):
            history.setdefault(pair, []).append(
                {
                    "team_a_id": row.home_team_id,
                    "team_b_id": row.away_team_id,
                    "goals_a": row.home_score,
                    "goals_b": row.away_score,
                }
            )

    return pd.DataFrame(rows, columns=["match_id"] + H2H_COLUMNS)
