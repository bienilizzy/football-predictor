"""Hand-computed checks for the feature engineering math (no DB required)."""
from __future__ import annotations

import pandas as pd
import pytest

from football_predictor.features.form import add_form_features
from football_predictor.features.h2h import compute_h2h_features
from football_predictor.features.referee_bias import MIN_REFEREE_MATCHES, compute_referee_features
from football_predictor.features.xg_features import add_xg_features


def test_form_features_no_leakage_and_rolling_average():
    log = pd.DataFrame(
        {
            "team_id": [1, 1, 1],
            "kickoff_utc": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
            "is_home": [True, True, True],
            "goals_for": [2, 0, 3],
            "goals_against": [1, 0, 1],
            "points": [3.0, 1.0, 3.0],
        }
    )

    out = add_form_features(log, windows=(2,))

    # Match 1: no prior matches -> NaN (never leaks the match's own result).
    assert pd.isna(out.loc[0, "form_pts_2"])

    # Match 2: only match 1 is "prior".
    assert out.loc[1, "form_pts_2"] == pytest.approx(3.0)
    assert out.loc[1, "form_gf_2"] == pytest.approx(2.0)
    assert out.loc[1, "form_ga_2"] == pytest.approx(1.0)

    # Match 3: average of matches 1 and 2.
    assert out.loc[2, "form_pts_2"] == pytest.approx((3.0 + 1.0) / 2)
    assert out.loc[2, "form_gf_2"] == pytest.approx((2.0 + 0.0) / 2)
    assert out.loc[2, "form_ga_2"] == pytest.approx((1.0 + 0.0) / 2)

    # All matches are home matches, so venue form matches overall form.
    assert out.loc[2, "venue_form_pts_2"] == pytest.approx(out.loc[2, "form_pts_2"])


def test_xg_overperformance_rolling():
    log = pd.DataFrame(
        {
            "team_id": [1, 1, 1],
            "kickoff_utc": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
            "goals_for": [2, 0, 3],
            "xg_for": [1.0, 1.5, 2.0],
            "xg_against": [0.5, 1.0, 1.0],
        }
    )

    out = add_xg_features(log, windows=(2,))

    assert pd.isna(out.loc[0, "xg_for_2"])
    assert out.loc[1, "xg_for_2"] == pytest.approx(1.0)
    # Match 1 overperformance = goals_for - xg_for = 2 - 1.0 = 1.0
    assert out.loc[1, "xg_overperf_2"] == pytest.approx(1.0)
    # Match 3 = average of match 1 (2 - 1.0 = 1.0) and match 2 (0 - 1.5 = -1.5)
    assert out.loc[2, "xg_overperf_2"] == pytest.approx((1.0 + (-1.5)) / 2)


def test_h2h_features_use_only_past_meetings():
    matches = pd.DataFrame(
        {
            "match_id": [1, 2, 3],
            "kickoff_utc": pd.to_datetime(["2023-01-01", "2023-06-01", "2024-01-01"]),
            "home_team_id": [1, 2, 1],
            "away_team_id": [2, 1, 2],
            "home_score": [2, 1, 0],
            "away_score": [1, 1, 0],
        }
    )

    out = compute_h2h_features(matches, last_n=5)

    # First-ever meeting: no history.
    assert out.loc[0, "h2h_matches_count"] == 0
    assert pd.isna(out.loc[0, "h2h_home_win_rate"])

    # Second meeting (roles reversed): one prior meeting where today's home
    # team (team 2) was the away side and lost 2-1.
    assert out.loc[1, "h2h_matches_count"] == 1
    assert out.loc[1, "h2h_home_win_rate"] == pytest.approx(0.0)
    assert out.loc[1, "h2h_away_win_rate"] == pytest.approx(1.0)

    # Third meeting: two prior meetings (one win, one draw) from team 1's
    # perspective as home team.
    assert out.loc[2, "h2h_matches_count"] == 2
    assert out.loc[2, "h2h_home_win_rate"] == pytest.approx(0.5)
    assert out.loc[2, "h2h_draw_rate"] == pytest.approx(0.5)
    assert out.loc[2, "h2h_away_win_rate"] == pytest.approx(0.0)


def test_referee_bias_fallback_then_known():
    n = MIN_REFEREE_MATCHES + 1
    matches = pd.DataFrame(
        {
            "match_id": list(range(1, n + 1)),
            "kickoff_utc": pd.date_range("2024-01-01", periods=n, freq="7D"),
            "referee_name": ["Ref A"] * n,
            "home_yellow": [2] * n,
            "away_yellow": [2] * n,
            "home_red": [0] * n,
            "away_red": [0] * n,
            "home_score": [1] * n,
            "away_score": [0] * n,
        }
    )

    out = compute_referee_features(matches)

    # Before the referee has officiated MIN_REFEREE_MATCHES games, fall back
    # to the league-wide average with a neutral (zero) bias delta.
    for i in range(MIN_REFEREE_MATCHES):
        assert out.loc[i, "referee_known"] == 0
        assert out.loc[i, "referee_home_win_rate_delta"] == pytest.approx(0.0)

    # Once the referee has MIN_REFEREE_MATCHES prior games, use their own stats.
    last = MIN_REFEREE_MATCHES
    assert out.loc[last, "referee_known"] == 1
    assert out.loc[last, "referee_avg_cards"] == pytest.approx(4.0)
    # Every prior match (referee's and league-wide) was a home win, so the
    # referee's home-win rate matches the league average -> delta of 0.
    assert out.loc[last, "referee_home_win_rate_delta"] == pytest.approx(0.0)
