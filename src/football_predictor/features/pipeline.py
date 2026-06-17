"""Builds the full match feature matrix and persists it to the MatchFeatures table.

Combines rolling form, rolling xG, head-to-head, referee bias, and weather
features into one wide row per match, keyed by match_id. Rows for SCHEDULED
matches get a feature vector (for prediction) but `target=None`; rows for
FINISHED matches get `target` in {"H", "D", "A"}.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from football_predictor.db.models import Match, MatchFeatures, MatchStats, WeatherRecord
from football_predictor.db.session import get_session
from football_predictor.features.form import DEFAULT_WINDOWS as FORM_WINDOWS
from football_predictor.features.form import add_form_features
from football_predictor.features.h2h import compute_h2h_features
from football_predictor.features.referee_bias import compute_referee_features
from football_predictor.features.weather_features import compute_weather_features
from football_predictor.features.xg_features import DEFAULT_WINDOWS as XG_WINDOWS
from football_predictor.features.xg_features import add_xg_features

logger = logging.getLogger(__name__)


def _load_matches_df(session) -> pd.DataFrame:
    matches = session.query(Match).order_by(Match.kickoff_utc).all()
    return pd.DataFrame(
        [
            {
                "match_id": m.id,
                "kickoff_utc": m.kickoff_utc,
                "home_team_id": m.home_team_id,
                "away_team_id": m.away_team_id,
                "home_score": m.home_score,
                "away_score": m.away_score,
                "status": m.status,
                "referee_name": m.referee_name,
                "home_yellow": m.home_yellow,
                "away_yellow": m.away_yellow,
                "home_red": m.home_red,
                "away_red": m.away_red,
            }
            for m in matches
        ]
    )


def _load_match_stats_df(session) -> pd.DataFrame:
    stats = session.query(MatchStats).all()
    return pd.DataFrame(
        [{"match_id": s.match_id, "team_id": s.team_id, "xg": s.xg, "xga": s.xga} for s in stats]
    )


def _load_weather_df(session) -> pd.DataFrame:
    records = session.query(WeatherRecord).all()
    return pd.DataFrame(
        [
            {"match_id": w.match_id, "temp_c": w.temp_c, "precip_mm": w.precip_mm, "wind_kph": w.wind_kph}
            for w in records
        ]
    )


def _build_team_log(matches: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team), used to compute rolling form/xG features."""
    home = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "team_id": matches["home_team_id"],
            "kickoff_utc": matches["kickoff_utc"],
            "is_home": True,
            "goals_for": matches["home_score"],
            "goals_against": matches["away_score"],
        }
    )
    away = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "team_id": matches["away_team_id"],
            "kickoff_utc": matches["kickoff_utc"],
            "is_home": False,
            "goals_for": matches["away_score"],
            "goals_against": matches["home_score"],
        }
    )
    log = pd.concat([home, away], ignore_index=True)

    log["points"] = np.select(
        [log["goals_for"] > log["goals_against"], log["goals_for"] == log["goals_against"]],
        [3, 1],
        default=0,
    ).astype(float)
    log.loc[log["goals_for"].isna(), "points"] = np.nan

    if not stats.empty:
        log = log.merge(stats, on=["match_id", "team_id"], how="left")
    else:
        log["xg"] = np.nan
        log["xga"] = np.nan
    log = log.rename(columns={"xg": "xg_for", "xga": "xg_against"})

    return log.sort_values(["team_id", "kickoff_utc"]).reset_index(drop=True)


def _team_feature_column_names() -> list[str]:
    cols = []
    for w in FORM_WINDOWS:
        cols += [
            f"form_pts_{w}",
            f"form_gf_{w}",
            f"form_ga_{w}",
            f"venue_form_pts_{w}",
            f"venue_form_gf_{w}",
            f"venue_form_ga_{w}",
        ]
    for w in XG_WINDOWS:
        cols += [f"xg_for_{w}", f"xg_against_{w}", f"xg_overperf_{w}"]
    return cols


def _team_log_to_match_features(log: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-team rolling features back to one row per match, prefixed home_/away_."""
    feature_cols = _team_feature_column_names()

    home = log.loc[log["is_home"], ["match_id", *feature_cols]].rename(
        columns={c: f"home_{c}" for c in feature_cols}
    )
    away = log.loc[~log["is_home"], ["match_id", *feature_cols]].rename(
        columns={c: f"away_{c}" for c in feature_cols}
    )
    return home.merge(away, on="match_id", how="outer")


def _result(home_score, away_score) -> str | None:
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def build_feature_matrix() -> pd.DataFrame:
    """Returns a DataFrame with one row per match: match_id, feature columns, target."""
    with get_session() as session:
        matches = _load_matches_df(session)
        stats = _load_match_stats_df(session)
        weather = _load_weather_df(session)

    if matches.empty:
        return pd.DataFrame()

    log = _build_team_log(matches, stats)
    log = add_form_features(log, FORM_WINDOWS)
    log = add_xg_features(log, XG_WINDOWS)

    feature_df = matches[["match_id"]].copy()
    feature_df = feature_df.merge(_team_log_to_match_features(log), on="match_id", how="left")
    feature_df = feature_df.merge(compute_h2h_features(matches), on="match_id", how="left")
    feature_df = feature_df.merge(compute_referee_features(matches), on="match_id", how="left")
    feature_df = feature_df.merge(compute_weather_features(matches, weather), on="match_id", how="left")

    feature_cols = [c for c in feature_df.columns if c != "match_id"]
    feature_df[feature_cols] = feature_df[feature_cols].fillna(0.0)

    feature_df["target"] = matches.apply(lambda r: _result(r["home_score"], r["away_score"]), axis=1).values

    return feature_df


def feature_columns(feature_df: pd.DataFrame) -> list[str]:
    return [c for c in feature_df.columns if c not in ("match_id", "target")]


def persist_feature_matrix(feature_df: pd.DataFrame) -> int:
    cols = feature_columns(feature_df)
    count = 0
    with get_session() as session:
        existing = {mf.match_id: mf for mf in session.query(MatchFeatures).all()}
        for _, row in feature_df.iterrows():
            match_id = int(row["match_id"])
            features = {c: float(row[c]) for c in cols}
            target = row["target"]

            mf = existing.get(match_id)
            if mf is None:
                mf = MatchFeatures(match_id=match_id, features=features, target=target)
                session.add(mf)
            else:
                mf.features = features
                mf.target = target
            count += 1
    return count


def run() -> int:
    feature_df = build_feature_matrix()
    if feature_df.empty:
        logger.warning("No matches in database; nothing to build features for.")
        return 0
    n = persist_feature_matrix(feature_df)
    logger.info("Persisted features for %d matches", n)
    return n
