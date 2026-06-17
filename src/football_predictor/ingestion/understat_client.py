"""xG / xGA per match from Understat, via the `understatapi` scraping client.

Understat has no official API; `understatapi` scrapes JSON embedded in the site's
pages. A single league-level request returns xG for every match in a season, which
keeps scraping volume low (one request per season rather than per match).
"""
from __future__ import annotations

import logging

from understatapi import UnderstatClient

from config.settings import settings
from football_predictor.db.models import Match, MatchStats, Team
from football_predictor.db.session import get_session
from football_predictor.ingestion.football_data_co_uk import season_code_to_label
from football_predictor.reference_data import understat_name_map

logger = logging.getLogger(__name__)


def season_code_to_understat_season(season_code: str) -> str:
    """'2324' -> '2023' (Understat seasons are keyed by the starting year)."""
    return f"20{season_code[:2]}"


def fetch_season_xg(season_code: str) -> list[dict]:
    understat_season = season_code_to_understat_season(season_code)
    with UnderstatClient() as understat:
        return understat.league(league=settings.understat_league).get_match_data(season=understat_season)


def _upsert_xg(session, match_id: int, team_id: int, is_home: bool, xg: float, xga: float) -> None:
    stats = session.query(MatchStats).filter_by(match_id=match_id, team_id=team_id).one_or_none()
    if stats is None:
        stats = MatchStats(match_id=match_id, team_id=team_id, is_home=is_home)
        session.add(stats)
    stats.xg = xg
    stats.xga = xga


def ingest_season_xg(season_code: str) -> int:
    """Fetch a season's match-level xG from Understat and merge into MatchStats."""
    season_label = season_code_to_label(season_code)
    name_map = understat_name_map()
    matches_data = fetch_season_xg(season_code)

    processed = 0
    skipped = 0
    with get_session() as session:
        teams_by_name = {t.canonical_name: t for t in session.query(Team).all()}

        for m in matches_data:
            if not m.get("isResult"):
                continue  # fixture not yet played

            home_canon = name_map.get(m["h"]["title"])
            away_canon = name_map.get(m["a"]["title"])
            home_team = teams_by_name.get(home_canon) if home_canon else None
            away_team = teams_by_name.get(away_canon) if away_canon else None
            if home_team is None or away_team is None:
                skipped += 1
                continue

            match = (
                session.query(Match)
                .filter_by(season=season_label, home_team_id=home_team.id, away_team_id=away_team.id)
                .one_or_none()
            )
            if match is None:
                # No corresponding football-data.co.uk row (e.g. season not ingested yet)
                skipped += 1
                continue

            home_xg = float(m["xG"]["h"])
            away_xg = float(m["xG"]["a"])

            session.flush()
            _upsert_xg(session, match.id, home_team.id, True, xg=home_xg, xga=away_xg)
            _upsert_xg(session, match.id, away_team.id, False, xg=away_xg, xga=home_xg)
            processed += 1

    if skipped:
        logger.warning("Understat season %s: skipped %d matches (unmapped team or no DB match)", season_code, skipped)
    return processed
