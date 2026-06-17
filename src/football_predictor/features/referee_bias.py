"""Referee bias features: card tendencies and home-advantage skew per referee.

Computed from football-data.co.uk's per-match cards (HY/AY/HR/AR) and results,
using only matches officiated *before* the current one. Referees with too little
prior history fall back to the league-wide average.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_REFEREE_MATCHES = 5

REFEREE_COLUMNS = ["referee_avg_cards", "referee_home_win_rate_delta", "referee_known"]


def compute_referee_features(matches: pd.DataFrame) -> pd.DataFrame:
    """`matches` needs: match_id, kickoff_utc, referee_name, home_yellow, away_yellow,
    home_red, away_red, home_score, away_score.

    Returns one row per match_id with the REFEREE_COLUMNS above.
    """
    df = matches.sort_values("kickoff_utc").reset_index(drop=True)

    ref_history: dict[str, list[dict]] = {}
    league_history: list[dict] = []
    rows: list[dict] = []

    for row in df.itertuples(index=False):
        ref_matches = ref_history.get(row.referee_name, []) if pd.notna(row.referee_name) else []

        if league_history:
            league_avg_cards = float(np.mean([m["cards"] for m in league_history]))
            league_home_win_rate = float(np.mean([m["home_win"] for m in league_history]))
        else:
            league_avg_cards = np.nan
            league_home_win_rate = np.nan

        if len(ref_matches) >= MIN_REFEREE_MATCHES:
            ref_avg_cards = float(np.mean([m["cards"] for m in ref_matches]))
            ref_home_win_rate = float(np.mean([m["home_win"] for m in ref_matches]))
            home_win_rate_delta = ref_home_win_rate - league_home_win_rate
            referee_known = 1
        else:
            ref_avg_cards = league_avg_cards
            home_win_rate_delta = 0.0
            referee_known = 0

        rows.append(
            {
                "match_id": row.match_id,
                "referee_avg_cards": ref_avg_cards,
                "referee_home_win_rate_delta": home_win_rate_delta,
                "referee_known": referee_known,
            }
        )

        if pd.notna(row.home_score) and pd.notna(row.away_score):
            cards = sum(
                v
                for v in (row.home_yellow, row.away_yellow, row.home_red, row.away_red)
                if pd.notna(v)
            )
            record = {"cards": float(cards), "home_win": 1.0 if row.home_score > row.away_score else 0.0}
            league_history.append(record)
            if pd.notna(row.referee_name):
                ref_history.setdefault(row.referee_name, []).append(record)

    return pd.DataFrame(rows, columns=["match_id"] + REFEREE_COLUMNS)
