"""Historical match results, referees, cards and shot stats from football-data.co.uk.

This is the free, no-API-key, no-rate-limit backbone of the training dataset. CSVs
are published per season per league, e.g.:

    https://www.football-data.co.uk/mmz4281/2425/E0.csv

Columns of interest (E0 = Premier League):
    Date, Time, HomeTeam, AwayTeam, FTHG, FTAG, FTR, Referee,
    HS, AS, HST, AST, HF, AF, HY, AY, HR, AR
"""
from __future__ import annotations

import logging

import httpx
import pandas as pd

from config.settings import settings
from football_predictor.db.models import Match, MatchStats, Team
from football_predictor.db.session import get_session
from football_predictor.reference_data import fd_co_uk_name_map

logger = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281"
DEFAULT_KICKOFF_TIME = "15:00"


def season_code_to_label(season_code: str) -> str:
    """'2324' -> '2023-2024'."""
    return f"20{season_code[:2]}-20{season_code[2:]}"


def label_to_season_code(label: str) -> str:
    """'2025-2026' -> '2526'."""
    start, end = label.split("-")
    return f"{start[2:]}{end[2:]}"


def _csv_path(season_code: str):
    out_dir = settings.data_raw_dir / "football_data_co_uk"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{settings.fd_co_uk_league_code}_{season_code}.csv"


def download_season(season_code: str, force: bool = False):
    """Download (and cache) the raw season CSV. Returns the local file path."""
    path = _csv_path(season_code)
    if path.exists() and not force:
        return path

    url = f"{BASE_URL}/{season_code}/{settings.fd_co_uk_league_code}.csv"
    logger.info("Downloading %s", url)
    # Uses httpx rather than requests: this host's TLS handshake is incompatible
    # with urllib3's default SSL context on some OpenSSL builds (SSLEOFError),
    # while httpx's handshake works fine.
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def _safe_int(value) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def load_season(season_code: str, force_download: bool = False) -> pd.DataFrame:
    """Download (if needed) and return a normalized per-match DataFrame for a season."""
    path = download_season(season_code, force=force_download)
    df = pd.read_csv(path, encoding="latin-1")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    dates = pd.to_datetime(df["Date"], dayfirst=True)
    if "Time" in df.columns:
        times = df["Time"].fillna(DEFAULT_KICKOFF_TIME)
    else:
        times = DEFAULT_KICKOFF_TIME
    kickoff = pd.to_datetime(dates.dt.strftime("%Y-%m-%d") + " " + times, errors="coerce")
    kickoff = kickoff.fillna(dates)

    out = pd.DataFrame(
        {
            "season": season_code_to_label(season_code),
            "kickoff_utc": kickoff,
            "home_team": df["HomeTeam"],
            "away_team": df["AwayTeam"],
            "home_score": df["FTHG"].astype(int),
            "away_score": df["FTAG"].astype(int),
            "referee": df.get("Referee"),
            "home_shots": df.get("HS"),
            "away_shots": df.get("AS"),
            "home_shots_on_target": df.get("HST"),
            "away_shots_on_target": df.get("AST"),
            "home_fouls": df.get("HF"),
            "away_fouls": df.get("AF"),
            "home_yellow": df.get("HY"),
            "away_yellow": df.get("AY"),
            "home_red": df.get("HR"),
            "away_red": df.get("AR"),
        }
    )
    return out.reset_index(drop=True)


def _upsert_match_stats(session, match_id: int, team_id: int, is_home: bool, shots, sot) -> None:
    stats = (
        session.query(MatchStats)
        .filter_by(match_id=match_id, team_id=team_id)
        .one_or_none()
    )
    if stats is None:
        stats = MatchStats(match_id=match_id, team_id=team_id, is_home=is_home)
        session.add(stats)
    stats.shots = _safe_int(shots)
    stats.shots_on_target = _safe_int(sot)


def ingest_season(season_code: str, force_download: bool = False) -> int:
    """Load a season's CSV and upsert Match + MatchStats rows. Returns rows processed."""
    df = load_season(season_code, force_download=force_download)
    name_map = fd_co_uk_name_map()

    processed = 0
    skipped = 0
    with get_session() as session:
        teams_by_name = {t.canonical_name: t for t in session.query(Team).all()}

        for _, row in df.iterrows():
            home_canon = name_map.get(row["home_team"])
            away_canon = name_map.get(row["away_team"])
            home_team = teams_by_name.get(home_canon) if home_canon else None
            away_team = teams_by_name.get(away_canon) if away_canon else None

            if home_team is None or away_team is None:
                skipped += 1
                logger.debug(
                    "Skipping unmapped fixture %s vs %s (%s)",
                    row["home_team"],
                    row["away_team"],
                    row["season"],
                )
                continue

            match = (
                session.query(Match)
                .filter_by(season=row["season"], home_team_id=home_team.id, away_team_id=away_team.id)
                .one_or_none()
            )
            if match is None:
                match = Match(
                    season=row["season"],
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                )
                session.add(match)

            match.kickoff_utc = row["kickoff_utc"].to_pydatetime()
            match.home_score = int(row["home_score"])
            match.away_score = int(row["away_score"])
            match.status = "FINISHED"
            match.referee_name = row["referee"] if pd.notna(row["referee"]) else None
            match.home_yellow = _safe_int(row["home_yellow"])
            match.away_yellow = _safe_int(row["away_yellow"])
            match.home_red = _safe_int(row["home_red"])
            match.away_red = _safe_int(row["away_red"])
            match.home_fouls = _safe_int(row["home_fouls"])
            match.away_fouls = _safe_int(row["away_fouls"])

            session.flush()  # ensure match.id is populated

            _upsert_match_stats(session, match.id, home_team.id, True, row["home_shots"], row["home_shots_on_target"])
            _upsert_match_stats(session, match.id, away_team.id, False, row["away_shots"], row["away_shots_on_target"])

            processed += 1

    if skipped:
        logger.warning("Season %s: skipped %d fixtures with unmapped teams", season_code, skipped)
    return processed
