"""Per-sport feature engineering for the multi-sport prediction pipeline.

`build_features` dispatches to a per-sport builder. Each builder takes a
normalized fixture/result DataFrame (see
`football_predictor.sports.data_layer`) and returns it with feature columns
added, reusing `home_team`/`away_team`/`datetime` (or `driver1`/`driver2` for
F1). Where the same underlying concept applies across sports, columns share a
name (e.g. `form_diff`) so downstream models can be built generically.

All rolling/history features are computed *pre-match* (via `shift(1)` /
`expanding`) so nothing leaks the outcome of the row it describes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from football_predictor.features import pipeline
from football_predictor.features.form import rolling_pre_match
from football_predictor.features.h2h import compute_h2h_features

FORM_WINDOW = 5
SETS_FORM_WINDOW = 5
VENUE_FORM_WINDOW = 10
VENUE_AVG_SCORE_DEFAULT = 300.0  # typical ODI innings total, used with no venue history

ELO_INITIAL = 1500.0
ELO_K = 32.0
FATIGUE_WINDOW_DAYS = 7


def _pairwise_rolling_diff(
    df: pd.DataFrame,
    p1_col: str,
    p2_col: str,
    p1_value: pd.Series,
    p2_value: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling pre-match average of a per-row value, grouped by entity, as p1 - p2.

    `df` (and `p1_value`/`p2_value`, which must share its row order) must
    already be sorted chronologically. `p1_col`/`p2_col` name the columns in
    `df` identifying the two entities (e.g. "home_team"/"away_team"). An
    entity that appears under both columns over time (e.g. a team that's
    sometimes home, sometimes away) has its history merged in chronological
    order before the rolling average is computed.
    """
    n = len(df)
    pos = np.arange(n)
    log = pd.concat(
        [
            pd.DataFrame({"entity": df[p1_col].values, "value": p1_value.values, "pos": pos, "role": 0, "row": df.index}),
            pd.DataFrame({"entity": df[p2_col].values, "value": p2_value.values, "pos": pos, "role": 1, "row": df.index}),
        ],
        ignore_index=True,
    )
    log = log.sort_values(["pos", "role"], kind="stable")
    log["rolling"] = log.groupby("entity")["value"].transform(lambda s: rolling_pre_match(s, window))
    p1_rolling = log.loc[log["role"] == 0].set_index("row")["rolling"]
    p2_rolling = log.loc[log["role"] == 1].set_index("row")["rolling"]
    return (p1_rolling - p2_rolling).reindex(df.index)


def football_features(df: pd.DataFrame) -> pd.DataFrame:
    """Football features via the existing DB-backed pipeline (features/pipeline.py).

    `df` is accepted for interface consistency with `build_features`; the
    feature matrix itself comes from `pipeline.build_feature_matrix()`
    (rolling form, xG, head-to-head, referee, weather - one row per match_id).
    Adds `form_diff` and `h2h_edge` for consistency with the other sports.
    """
    feature_df = pipeline.build_feature_matrix()
    if feature_df.empty:
        return feature_df

    feature_df["form_diff"] = feature_df["home_form_pts_5"] - feature_df["away_form_pts_5"]
    feature_df["h2h_edge"] = feature_df["h2h_home_win_rate"] - feature_df["h2h_away_win_rate"]
    return feature_df


def cricket_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cricket features: recent form, head-to-head, venue average score.

    Expects `home_team`, `away_team`, `datetime` (always present). For
    completed matches, `home_score`/`away_score` (runs) enable form/H2H/venue
    history; `venue` enables per-venue scoring history. Missing inputs
    default to neutral values so this never raises on fixture-only data.

    Adds: form_diff (recent win-rate diff), h2h_edge (head-to-head win-rate
    diff), venue_avg_score (recent average total runs at the venue).
    """
    out = df.reset_index(drop=True).copy()
    ordered = out.sort_values("datetime")

    has_scores = {"home_score", "away_score"}.issubset(ordered.columns)
    if has_scores:
        home_score = ordered["home_score"].astype(float)
        away_score = ordered["away_score"].astype(float)
        home_won = (home_score > away_score).astype(float)
        away_won = (away_score > home_score).astype(float)
    else:
        home_score = away_score = pd.Series(np.nan, index=ordered.index)
        home_won = away_won = pd.Series(np.nan, index=ordered.index)

    out["form_diff"] = _pairwise_rolling_diff(
        ordered, "home_team", "away_team", home_won, away_won, FORM_WINDOW
    ).fillna(0.0)

    h2h_input = pd.DataFrame(
        {
            "match_id": ordered.index,
            "kickoff_utc": ordered["datetime"],
            "home_team_id": ordered["home_team"],
            "away_team_id": ordered["away_team"],
            "home_score": home_score,
            "away_score": away_score,
        }
    )
    h2h = compute_h2h_features(h2h_input).set_index("match_id")
    h2h_edge = h2h["h2h_home_win_rate"] - h2h["h2h_away_win_rate"]
    out["h2h_edge"] = h2h_edge.reindex(out.index).fillna(0.0)

    if has_scores:
        total_runs = home_score + away_score
        if "venue" in ordered.columns:
            venue_log = pd.DataFrame(
                {"venue": ordered["venue"].values, "total": total_runs.values}, index=ordered.index
            )
            venue_avg = venue_log.groupby("venue")["total"].transform(
                lambda s: rolling_pre_match(s, VENUE_FORM_WINDOW)
            )
        else:
            venue_avg = rolling_pre_match(total_runs.reset_index(drop=True), VENUE_FORM_WINDOW)
            venue_avg.index = ordered.index
        out["venue_avg_score"] = venue_avg.reindex(out.index)
    else:
        out["venue_avg_score"] = np.nan
    out["venue_avg_score"] = out["venue_avg_score"].fillna(VENUE_AVG_SCORE_DEFAULT)

    return out


def _surface_elo_diff(df: pd.DataFrame, home_won: pd.Series, surfaces: pd.Series, k: float = ELO_K) -> pd.Series:
    """Pre-match surface-specific Elo difference (home - away), scaled to roughly [-1, 1].

    Updates a per-(player, surface) Elo rating sequentially in chronological
    order (`df` must already be sorted by time); unrated players start at
    ELO_INITIAL. Matches with an unknown outcome (`home_won` is NaN) are read
    but don't update ratings.
    """
    ratings: dict[tuple[str, str], float] = {}
    diffs = []
    for (_, row), surface, won in zip(df.iterrows(), surfaces, home_won):
        home_key = (row["home_team"], surface)
        away_key = (row["away_team"], surface)
        home_elo = ratings.get(home_key, ELO_INITIAL)
        away_elo = ratings.get(away_key, ELO_INITIAL)
        diffs.append((home_elo - away_elo) / 400.0)

        if pd.notna(won):
            expected_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400.0))
            ratings[home_key] = home_elo + k * (won - expected_home)
            ratings[away_key] = away_elo + k * ((1.0 - won) - (1.0 - expected_home))

    return pd.Series(diffs, index=df.index)


def _fatigue_diff(df: pd.DataFrame, window_days: int = FATIGUE_WINDOW_DAYS) -> pd.Series:
    """Pre-match fatigue diff: matches played by the away player minus the home
    player in the preceding `window_days`. Positive means the home player is
    fresher (an advantage)."""
    times = pd.to_datetime(df["datetime"])
    history: dict[str, list[pd.Timestamp]] = {}
    diffs = []
    for player_home, player_away, t in zip(df["home_team"], df["away_team"], times):
        cutoff = t - pd.Timedelta(days=window_days)
        home_recent = sum(1 for ts in history.get(player_home, []) if ts > cutoff)
        away_recent = sum(1 for ts in history.get(player_away, []) if ts > cutoff)
        diffs.append(float(away_recent - home_recent))
        history.setdefault(player_home, []).append(t)
        history.setdefault(player_away, []).append(t)
    return pd.Series(diffs, index=df.index)


def tennis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tennis features: surface-specific Elo, recent sets win%, fatigue.

    Expects `home_team`, `away_team` (players), `datetime` (always present).
    `surface` (e.g. "hard"/"clay"/"grass") enables surface-specific Elo;
    `home_sets_won`/`away_sets_won` (completed matches) enable Elo updates and
    recent sets win% history. Missing inputs default to neutral values.

    Adds: surface_advantage (surface Elo diff), form_diff (recent sets win%
    diff), fatigue_diff (recent match-load diff).
    """
    out = df.reset_index(drop=True).copy()
    ordered = out.sort_values("datetime")

    has_sets = {"home_sets_won", "away_sets_won"}.issubset(ordered.columns)
    surfaces = ordered["surface"] if "surface" in ordered.columns else pd.Series("overall", index=ordered.index)

    if has_sets:
        home_sets = ordered["home_sets_won"].astype(float)
        away_sets = ordered["away_sets_won"].astype(float)
        total_sets = home_sets + away_sets
        home_pct = (home_sets / total_sets).where(total_sets > 0)
        away_pct = (away_sets / total_sets).where(total_sets > 0)
        home_won = (home_sets > away_sets).astype(float)
    else:
        home_pct = away_pct = pd.Series(np.nan, index=ordered.index)
        home_won = pd.Series(np.nan, index=ordered.index)

    out["surface_advantage"] = _surface_elo_diff(ordered, home_won, surfaces).reindex(out.index).fillna(0.0)
    out["form_diff"] = _pairwise_rolling_diff(
        ordered, "home_team", "away_team", home_pct, away_pct, SETS_FORM_WINDOW
    ).fillna(0.0)
    out["fatigue_diff"] = _fatigue_diff(ordered).reindex(out.index).fillna(0.0)

    return out


def _grid_diff(df: pd.DataFrame, col1: str, col2: str, default: float = 0.0) -> pd.Series:
    """`col2` - `col1`, defaulting to `default` when either column is missing/NaN.

    For grid/penalty positions (lower = better), a positive result means
    driver1 has the more favorable (lower) value.
    """
    if col1 not in df.columns or col2 not in df.columns:
        return pd.Series(default, index=df.index)
    return (df[col2].astype(float) - df[col1].astype(float)).fillna(default)


def _track_history_diff(df: pd.DataFrame, finish1: pd.Series, finish2: pd.Series) -> pd.Series:
    """Pre-race average finishing position at this circuit, driver2 - driver1.

    Averages over all prior races at the circuit (`df` must already be sorted
    by time), merging a driver's history in chronological order regardless of
    whether they raced as driver1 or driver2. Positive means driver1's
    history at this circuit is better (lower average finishing position).
    """
    n = len(df)
    pos = np.arange(n)
    log = pd.concat(
        [
            pd.DataFrame(
                {"driver": df["driver1"].values, "circuit": df["circuit"].values, "finish": finish1.values, "pos": pos, "role": 0, "row": df.index}
            ),
            pd.DataFrame(
                {"driver": df["driver2"].values, "circuit": df["circuit"].values, "finish": finish2.values, "pos": pos, "role": 1, "row": df.index}
            ),
        ],
        ignore_index=True,
    )
    log = log.sort_values(["pos", "role"], kind="stable")
    log["history"] = log.groupby(["driver", "circuit"])["finish"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    h1 = log.loc[log["role"] == 0].set_index("row")["history"]
    h2 = log.loc[log["role"] == 1].set_index("row")["history"]
    return (h2 - h1).reindex(df.index)


def f1_features(df: pd.DataFrame) -> pd.DataFrame:
    """F1 features: qualifying position, track history, engine penalty.

    Expects `driver1`, `driver2`, `datetime` (always present).
    `quali_pos_driver1`/`quali_pos_driver2` (grid slot, lower = better) and
    `engine_penalty_driver1`/`engine_penalty_driver2` (grid penalty places)
    enable their respective diffs; `circuit` plus
    `finish_pos_driver1`/`finish_pos_driver2` (race results) enable track
    history and recent-form. Missing inputs default to neutral values.

    Adds: quali_diff, engine_penalty_diff, track_history_diff, and form_diff
    (recent finishing-position diff) for consistency with the other sports.
    """
    out = df.reset_index(drop=True).copy()
    ordered = out.sort_values("datetime")

    out["quali_diff"] = _grid_diff(ordered, "quali_pos_driver1", "quali_pos_driver2").reindex(out.index)
    out["engine_penalty_diff"] = _grid_diff(ordered, "engine_penalty_driver1", "engine_penalty_driver2").reindex(
        out.index
    )

    has_finish = {"finish_pos_driver1", "finish_pos_driver2"}.issubset(ordered.columns)
    if has_finish:
        finish1 = ordered["finish_pos_driver1"].astype(float)
        finish2 = ordered["finish_pos_driver2"].astype(float)
    else:
        finish1 = finish2 = pd.Series(np.nan, index=ordered.index)

    # avg_finish_driver2 - avg_finish_driver1: positive => driver1 has been
    # finishing higher (better, i.e. lower position numbers) recently.
    out["form_diff"] = _pairwise_rolling_diff(
        ordered, "driver2", "driver1", finish2, finish1, FORM_WINDOW
    ).fillna(0.0)

    if has_finish and "circuit" in ordered.columns:
        out["track_history_diff"] = _track_history_diff(ordered, finish1, finish2).fillna(0.0)
    else:
        out["track_history_diff"] = 0.0

    return out


def build_features(dataframe: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Dispatch to the per-sport feature builder for `sport`.

    `dataframe` is the normalized fixture/result data for `sport` (see
    `football_predictor.sports.data_layer.MultiSportDataFetcher`).
    """
    if sport == "football":
        return football_features(dataframe)
    elif sport == "cricket":
        return cricket_features(dataframe)
    elif sport == "tennis":
        return tennis_features(dataframe)
    elif sport == "f1":
        return f1_features(dataframe)
    else:
        raise ValueError(f"Unsupported sport: {sport!r}. Supported: football, cricket, tennis, f1")
