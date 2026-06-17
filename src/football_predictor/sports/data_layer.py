"""Multi-sport fixture/result data layer backed by the Sportmonks API.

Sportmonks v3 exposes a near-identical REST shape across sports
(`/v3/{sport}/{endpoint}/between/{date_from}/{date_to}`, paginated via
`pagination.has_more`). Football additionally falls back to
football-data.org (see `football_predictor.ingestion.football_data_org`) if
Sportmonks is unavailable, since that's the existing fixture source for the
Premier League.

Requires a free Sportmonks API key: https://www.sportmonks.com/
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
import time
from pathlib import Path

import requests

from config.settings import settings
from football_predictor.ingestion.football_data_org import FootballDataOrgClient

logger = logging.getLogger(__name__)

SPORTMONKS_BASE_URL = "https://api.sportmonks.com/v3"
CACHE_TTL_SECONDS = 60 * 60  # 1 hour
MAX_RETRIES = 3

# Per-sport Sportmonks API path segment, fixture-list endpoint name, and the
# participant role names used in the normalized schema. Formula 1 fixtures
# are races rather than team-vs-team matches, so their two participants are
# exposed as driver1/driver2 (the first two entries Sportmonks returns for
# the race, e.g. front-row grid order) instead of home_team/away_team.
SPORT_CONFIG = {
    "football": {"path": "football", "endpoint": "fixtures", "participant_keys": ("home_team", "away_team")},
    "cricket": {"path": "cricket", "endpoint": "fixtures", "participant_keys": ("home_team", "away_team")},
    "tennis": {"path": "tennis", "endpoint": "fixtures", "participant_keys": ("home_team", "away_team")},
    "f1": {"path": "formula1", "endpoint": "races", "participant_keys": ("driver1", "driver2")},
}


class SportmonksError(RuntimeError):
    """Raised when a Sportmonks API request fails after retries."""


class _FixtureCache:
    """SQLite-backed cache for normalized fixture/result lists, TTL = 1 hour."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # `nolock=1` + journal_mode=MEMORY is required for SQLite to work on
        # network-mounted filesystems (e.g. WSL's \\wsl.localhost UNC paths
        # accessed from Windows) - see config/settings.py:database_url.
        self._conn = sqlite3.connect(f"file:{db_path.as_posix()}?nolock=1", uri=True, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=MEMORY")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fixture_cache (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                fetched_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, cache_key: str, ttl_seconds: int = CACHE_TTL_SECONDS) -> list[dict] | None:
        row = self._conn.execute(
            "SELECT payload, fetched_at FROM fixture_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        payload, fetched_at = row
        if time.time() - fetched_at >= ttl_seconds:
            return None
        return json.loads(payload)

    def set(self, cache_key: str, payload: list[dict]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO fixture_cache (cache_key, payload, fetched_at) VALUES (?, ?, ?)",
            (cache_key, json.dumps(payload), time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _parse_datetime(value: str) -> str:
    """Normalize a Sportmonks/football-data.org timestamp to an ISO-8601 UTC string."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return dt.datetime.strptime(value, fmt).replace(tzinfo=dt.timezone.utc).isoformat()
        except ValueError:
            continue
    return value


def _normalize_sportmonks_fixture(sport: str, raw: dict) -> dict:
    """Map a raw Sportmonks fixture/race to the common schema."""
    p1_key, p2_key = SPORT_CONFIG[sport]["participant_keys"]
    participants = raw.get("participants") or []
    p1_name = participants[0]["name"] if len(participants) > 0 else None
    p2_name = participants[1]["name"] if len(participants) > 1 else None

    return {
        p1_key: p1_name,
        p2_key: p2_name,
        "datetime": _parse_datetime(raw.get("starting_at", "")),
        "sport_type": sport,
        "external_id": f"sportmonks:{raw.get('id')}",
    }


def _normalize_fd_org_fixture(raw: dict) -> dict:
    """Map a raw football-data.org match to the common schema."""
    return {
        "home_team": raw["homeTeam"]["name"],
        "away_team": raw["awayTeam"]["name"],
        "datetime": _parse_datetime(raw["utcDate"]),
        "sport_type": "football",
        "external_id": f"fd_org:{raw['id']}",
    }


class MultiSportDataFetcher:
    """Fetches and normalizes fixtures/results across sports via Sportmonks.

    Football falls back to football-data.org (`FootballDataOrgClient`) if the
    Sportmonks request fails, since that's the existing source of PL fixtures.
    """

    def __init__(self, api_key: str | None = None, cache_db_path: Path | None = None):
        self.api_key = api_key if api_key is not None else settings.sportmonks_api_key
        self.tier = settings.sportmonks_tier
        self.session = requests.Session()
        self.cache = _FixtureCache(cache_db_path or settings.db_file_path.parent / "sports_cache.db")

    def _get(self, path: str, params: dict | None = None) -> dict:
        if not self.api_key:
            raise SportmonksError(
                "SPORTMONKS_API_KEY is not set. Get a free key at "
                "https://www.sportmonks.com/ and add it to .env"
            )
        url = f"{SPORTMONKS_BASE_URL}{path}"
        params = dict(params or {})
        params["api_token"] = self.api_key

        for attempt in range(MAX_RETRIES):
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "5"))
                logger.warning("Rate limited by Sportmonks, waiting %ds", wait)
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                raise SportmonksError(f"Sportmonks {resp.status_code} on {path}: {resp.text[:200]}")
            return resp.json()
        raise SportmonksError("Sportmonks rate limit retries exhausted")

    def _get_paginated(self, path: str, params: dict | None = None) -> list[dict]:
        results: list[dict] = []
        page = 1
        while True:
            data = self._get(path, params={**(params or {}), "page": page})
            results.extend(data.get("data", []))
            if not data.get("pagination", {}).get("has_more"):
                break
            page += 1
        return results

    def fetch_fixtures(self, sport: str, date_from: dt.date, date_to: dt.date) -> list[dict]:
        """Return normalized fixtures for `sport` between `date_from` and `date_to` (inclusive).

        Results are cached in SQLite for 1 hour, keyed by sport + date range.
        Football falls back to football-data.org if Sportmonks fails.
        """
        if sport not in SPORT_CONFIG:
            raise ValueError(f"Unsupported sport: {sport!r}. Supported: {sorted(SPORT_CONFIG)}")

        cache_key = f"fixtures:{sport}:{date_from.isoformat()}:{date_to.isoformat()}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        config = SPORT_CONFIG[sport]
        try:
            raw_fixtures = self._get_paginated(
                f"/{config['path']}/{config['endpoint']}/between/{date_from.isoformat()}/{date_to.isoformat()}",
                params={"include": "participants"},
            )
            fixtures = [_normalize_sportmonks_fixture(sport, raw) for raw in raw_fixtures]
        except (SportmonksError, requests.RequestException) as exc:
            if sport != "football":
                raise
            logger.warning("Sportmonks fixtures request failed (%s); falling back to football-data.org", exc)
            client = FootballDataOrgClient()
            matches = client.fetch_matches(date_from=date_from, date_to=date_to)
            fixtures = [_normalize_fd_org_fixture(raw) for raw in matches]

        self.cache.set(cache_key, fixtures)
        return fixtures

    def fetch_historical(self, sport: str, seasons: list[int | str]) -> list[dict]:
        """Return normalized historical fixtures/results for `sport` across `seasons`.

        `seasons` are Sportmonks season IDs. Results are cached in SQLite for 1
        hour per (sport, season).
        """
        if sport not in SPORT_CONFIG:
            raise ValueError(f"Unsupported sport: {sport!r}. Supported: {sorted(SPORT_CONFIG)}")

        config = SPORT_CONFIG[sport]
        all_fixtures: list[dict] = []

        for season in seasons:
            cache_key = f"historical:{sport}:{season}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                all_fixtures.extend(cached)
                continue

            try:
                season_data = self._get(f"/{config['path']}/seasons/{season}")["data"]
                date_from = dt.date.fromisoformat(season_data["starting_at"][:10])
                date_to = dt.date.fromisoformat(season_data["ending_at"][:10])
                raw_fixtures = self._get_paginated(
                    f"/{config['path']}/{config['endpoint']}/between/{date_from.isoformat()}/{date_to.isoformat()}",
                    params={"include": "participants"},
                )
                fixtures = [_normalize_sportmonks_fixture(sport, raw) for raw in raw_fixtures]
            except (SportmonksError, requests.RequestException) as exc:
                logger.warning("Sportmonks historical request failed for %s season %s: %s", sport, season, exc)
                fixtures = []

            self.cache.set(cache_key, fixtures)
            all_fixtures.extend(fixtures)

        return all_fixtures

    def close(self) -> None:
        self.cache.close()
