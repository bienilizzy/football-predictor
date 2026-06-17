"""Upcoming fixture schedule from the football-data.org v4 API.

Used only for fixture scheduling (dates/kickoff times/matchday) since the free
tier has limited historical depth and a 10 requests/minute rate limit. Historical
results come from football-data.co.uk instead (see football_data_co_uk.py).

Requires a free API key: https://www.football-data.org/client/register
"""
from __future__ import annotations

import datetime as dt
import logging
import time

import requests

from config.settings import settings
from football_predictor.db.models import Match, OtherLeagueFixture, Team
from football_predictor.db.session import get_session
from football_predictor.reference_data import fd_org_name_map

logger = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"
MAX_RETRIES = 3

# Leagues to check for upcoming fixtures when the primary competition
# (`settings.fd_org_competition_code`, normally PL) has none scheduled - e.g.
# during its summer off-season. Matched by (case-insensitive) substring
# against the `name` field of `/v4/competitions`, since football-data.org's
# competition coverage depends on the API plan and codes aren't guaranteed.
FALLBACK_LEAGUE_NAMES = ["Allsvenskan", "Eliteserien", "J1 League"]


class FootballDataOrgClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key if api_key is not None else settings.football_data_org_api_key
        if not self.api_key:
            raise RuntimeError(
                "FOOTBALL_DATA_ORG_API_KEY is not set. Get a free key at "
                "https://www.football-data.org/client/register and add it to .env"
            )
        self.session = requests.Session()
        self.session.headers.update({"X-Auth-Token": self.api_key})

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{BASE_URL}{path}"
        for attempt in range(MAX_RETRIES):
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "60"))
                logger.warning("Rate limited by football-data.org, waiting %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("football-data.org rate limit retries exhausted")

    def fetch_matches(
        self,
        date_from: dt.date | None = None,
        date_to: dt.date | None = None,
        status: str | None = None,
        competition_code: str | None = None,
    ) -> list[dict]:
        params: dict = {}
        if date_from:
            params["dateFrom"] = date_from.isoformat()
        if date_to:
            params["dateTo"] = date_to.isoformat()
        if status:
            params["status"] = status
        code = competition_code or settings.fd_org_competition_code
        data = self._get(f"/competitions/{code}/matches", params=params)
        return data.get("matches", [])

    def fetch_upcoming_fixtures(self, days_ahead: int = 14, competition_code: str | None = None) -> list[dict]:
        today = dt.date.today()
        return self.fetch_matches(
            date_from=today,
            date_to=today + dt.timedelta(days=days_ahead),
            status="SCHEDULED",
            competition_code=competition_code,
        )

    def fetch_finished_matches(self, days_back: int = 7) -> list[dict]:
        today = dt.date.today()
        return self.fetch_matches(
            date_from=today - dt.timedelta(days=days_back), date_to=today, status="FINISHED"
        )

    def fetch_competitions(self) -> list[dict]:
        """All competitions available to this API key (`/v4/competitions`)."""
        data = self._get("/competitions")
        return data.get("competitions", [])


def _parse_kickoff(utc_date: str) -> dt.datetime:
    return dt.datetime.strptime(utc_date, "%Y-%m-%dT%H:%M:%SZ")


def _ingest_matches(matches: list[dict], status: str) -> int:
    name_map = fd_org_name_map()
    processed = 0

    with get_session() as session:
        teams_by_name = {t.canonical_name: t for t in session.query(Team).all()}

        for m in matches:
            home_canon = name_map.get(m["homeTeam"]["name"])
            away_canon = name_map.get(m["awayTeam"]["name"])
            home_team = teams_by_name.get(home_canon) if home_canon else None
            away_team = teams_by_name.get(away_canon) if away_canon else None

            if home_team is None or away_team is None:
                logger.warning(
                    "Skipping fd.org fixture with unmapped team(s): %s vs %s",
                    m["homeTeam"]["name"],
                    m["awayTeam"]["name"],
                )
                continue

            match = session.query(Match).filter_by(fd_org_id=m["id"]).one_or_none()
            if match is None:
                # Fall back to season+fixture lookup in case the row already exists
                # from a historical CSV (unlikely for upcoming/recent fixtures).
                match = (
                    session.query(Match)
                    .filter_by(
                        season=settings.current_season,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                    )
                    .one_or_none()
                )
            if match is None:
                match = Match(
                    season=settings.current_season,
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                )
                session.add(match)

            match.fd_org_id = m["id"]
            match.kickoff_utc = _parse_kickoff(m["utcDate"])
            match.matchday = m.get("matchday")
            match.status = status

            score = m.get("score", {}).get("fullTime", {})
            if score.get("home") is not None:
                match.home_score = score["home"]
            if score.get("away") is not None:
                match.away_score = score["away"]

            processed += 1

    return processed


def ingest_upcoming_fixtures(days_ahead: int = 14) -> int:
    client = FootballDataOrgClient()
    matches = client.fetch_upcoming_fixtures(days_ahead=days_ahead)
    return _ingest_matches(matches, status="SCHEDULED")


def ingest_recent_results(days_back: int = 7) -> int:
    client = FootballDataOrgClient()
    matches = client.fetch_finished_matches(days_back=days_back)
    return _ingest_matches(matches, status="FINISHED")


def _find_competition_code(competitions: list[dict], name_contains: str) -> str | None:
    needle = name_contains.casefold()
    for comp in competitions:
        if needle in comp.get("name", "").casefold():
            return comp.get("code")
    return None


def _ingest_other_league_fixtures(matches: list[dict], competition_code: str, competition_name: str) -> int:
    processed = 0

    with get_session() as session:
        for m in matches:
            fixture = session.query(OtherLeagueFixture).filter_by(fd_org_id=m["id"]).one_or_none()
            if fixture is None:
                fixture = OtherLeagueFixture(fd_org_id=m["id"])
                session.add(fixture)

            fixture.competition_code = competition_code
            fixture.competition_name = competition_name
            fixture.season = str(m.get("season", {}).get("startDate", ""))[:4] or None
            fixture.home_team_name = m["homeTeam"]["name"]
            fixture.away_team_name = m["awayTeam"]["name"]
            fixture.kickoff_utc = _parse_kickoff(m["utcDate"])
            fixture.status = m.get("status", "SCHEDULED")

            score = m.get("score", {}).get("fullTime", {})
            fixture.home_score = score.get("home")
            fixture.away_score = score.get("away")

            processed += 1

    return processed


def ingest_fallback_fixtures(days_ahead: int = 14) -> dict[str, int]:
    """Fetch upcoming fixtures from `FALLBACK_LEAGUE_NAMES`.

    Intended for when the primary competition (PL) has no scheduled matches
    in the requested window, e.g. during its summer off-season. Each league
    is looked up by name in `/v4/competitions` (since not every competition
    is available on every API plan); leagues that aren't found, or whose
    fixtures can't be fetched, are skipped with a logged warning. Returns the
    number of fixtures stored per league name that *was* found.
    """
    client = FootballDataOrgClient()
    results: dict[str, int] = {}

    try:
        competitions = client.fetch_competitions()
    except requests.HTTPError:
        logger.warning("Could not list football-data.org competitions for fallback leagues")
        return results

    for league_name in FALLBACK_LEAGUE_NAMES:
        code = _find_competition_code(competitions, league_name)
        if code is None:
            logger.info("Fallback league %r not available on this football-data.org plan", league_name)
            continue

        try:
            matches = client.fetch_upcoming_fixtures(days_ahead=days_ahead, competition_code=code)
        except requests.HTTPError:
            logger.warning("Could not fetch fixtures for fallback league %s (%s)", league_name, code)
            continue

        results[league_name] = _ingest_other_league_fixtures(matches, code, league_name)
        logger.info("Fallback league %s (%s): %d fixtures", league_name, code, results[league_name])

    return results
