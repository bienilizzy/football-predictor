"""Resolve SportPrediction outcomes once fixtures have been played.

For each `SportPrediction` whose `actual_outcome` is still NULL and whose
`kickoff_utc` is in the past, fetches the actual result and writes
`actual_outcome` (H / D / A) plus `correct` (bool) so the leaderboard and
calibration endpoints have data to display.

Fetch strategy by external_id prefix:
  ``fd_org:{id}``      — football only; calls football-data.org /v4/matches/{id}
  ``sportmonks:{id}``  — any sport; calls Sportmonks /v3/{sport}/fixtures/{id}
                         with ``?include=scores`` (football/cricket) or
                         ``?include=participants`` (tennis/f1).

Outcome mapping:
  - Football / Cricket: home_score vs away_score → H / D / A
  - Tennis: home_sets_won vs away_sets_won → H / A  (no D)
  - F1: finishing position of driver1 vs driver2 → H / A  (no D)
"""
from __future__ import annotations

import datetime as dt
import logging
import time

import requests

from config.settings import settings
from football_predictor.db.models import SportPrediction
from football_predictor.db.session import get_session
from football_predictor.sports.data_layer import SPORT_CONFIG

logger = logging.getLogger(__name__)

_FD_ORG_BASE = "https://api.football-data.org/v4"
_SPORTMONKS_BASE = "https://api.sportmonks.com/v3"
_MAX_RETRIES = 3
_RATE_LIMIT_SLEEP = 60


def _fd_org_result(fixture_id: str, api_key: str) -> str | None:
    """Return H / D / A for a finished football-data.org match, or None."""
    url = f"{_FD_ORG_BASE}/matches/{fixture_id}"
    for attempt in range(_MAX_RETRIES):
        resp = requests.get(url, headers={"X-Auth-Token": api_key}, timeout=30)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", _RATE_LIMIT_SLEEP))
            logger.warning("fd_org rate limit; sleeping %ds", wait)
            time.sleep(wait)
            continue
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "FINISHED":
            return None
        score = data.get("score", {}).get("fullTime", {})
        home = score.get("home")
        away = score.get("away")
        if home is None or away is None:
            return None
        if home > away:
            return "H"
        if away > home:
            return "A"
        return "D"
    return None


def _sportmonks_result(sport: str, fixture_id: str, api_key: str) -> str | None:
    """Return H / D / A for a Sportmonks fixture, or None."""
    config = SPORT_CONFIG[sport]
    path_segment = config["path"]
    endpoint = "fixtures" if sport != "f1" else "races"

    include = "scores" if sport in ("football", "cricket") else "participants"
    url = f"{_SPORTMONKS_BASE}/{path_segment}/{endpoint}/{fixture_id}"

    for attempt in range(_MAX_RETRIES):
        resp = requests.get(
            url,
            params={"api_token": api_key, "include": include},
            timeout=30,
        )
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", "5"))
            logger.warning("Sportmonks rate limit; sleeping %ds", wait)
            time.sleep(wait)
            continue
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json().get("data", {})
        return _parse_sportmonks_result(sport, data)
    return None


def _parse_sportmonks_result(sport: str, data: dict) -> str | None:
    """Parse a Sportmonks fixture/race payload into H / D / A."""
    if sport in ("football", "cricket"):
        # Sportmonks v3 football: data["scores"] is a list of score objects.
        # Each has "description" and "score" with "participant" ("home"/"away")
        # and "goals" (football) / "runs" (cricket). We want the "CURRENT"
        # description which reflects the final score.
        scores = data.get("scores") or []
        current = [s for s in scores if s.get("description") == "CURRENT"]
        home_score = next((s["score"].get("goals", s["score"].get("runs")) for s in current if s.get("score", {}).get("participant") == "home"), None)
        away_score = next((s["score"].get("goals", s["score"].get("runs")) for s in current if s.get("score", {}).get("participant") == "away"), None)
        if home_score is None or away_score is None:
            return None
        if home_score > away_score:
            return "H"
        if away_score > home_score:
            return "A"
        return "D"

    if sport == "tennis":
        # Tennis fixtures: participants list with "result" → "winner" (true/false)
        # or we count sets from scores. Try the winner flag first (simpler).
        participants = data.get("participants") or []
        p1_key, p2_key = SPORT_CONFIG["tennis"]["participant_keys"]
        for i, participant in enumerate(participants[:2]):
            result = participant.get("result") or {}
            if result.get("winner"):
                return "H" if i == 0 else "A"
        return None

    if sport == "f1":
        # F1 races: participants include "position" after the race. driver1 is
        # the first participant in the fixture, driver2 the second.
        participants = data.get("participants") or []
        if len(participants) < 2:
            return None
        pos1 = participants[0].get("result", {}).get("position")
        pos2 = participants[1].get("result", {}).get("position")
        if pos1 is None or pos2 is None:
            return None
        # Lower position number = better finish.
        return "H" if pos1 < pos2 else "A"

    return None


def resolve_sport_predictions(dry_run: bool = False) -> dict:
    """Resolve all unresolved past SportPredictions. Returns a summary dict.

    Args:
        dry_run: If True, fetch results but don't write them to the DB.
    """
    fd_org_key = settings.football_data_org_api_key
    sportmonks_key = settings.sportmonks_api_key
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)

    with get_session() as session:
        unresolved = (
            session.query(SportPrediction)
            .filter(
                SportPrediction.actual_outcome.is_(None),
                SportPrediction.kickoff_utc < now,
            )
            .order_by(SportPrediction.kickoff_utc)
            .all()
        )

    if not unresolved:
        logger.info("No unresolved past sport predictions to score")
        return {"resolved": 0, "skipped": 0, "errors": 0}

    resolved = skipped = errors = 0

    for row in unresolved:
        prefix, _, fixture_id = row.external_id.partition(":")
        actual: str | None = None

        try:
            if prefix == "fd_org":
                if not fd_org_key:
                    logger.debug("Skipping fd_org:%s — FOOTBALL_DATA_ORG_API_KEY not set", fixture_id)
                    skipped += 1
                    continue
                actual = _fd_org_result(fixture_id, fd_org_key)
            elif prefix == "sportmonks":
                if not sportmonks_key:
                    logger.debug("Skipping sportmonks:%s — SPORTMONKS_API_KEY not set", fixture_id)
                    skipped += 1
                    continue
                actual = _sportmonks_result(row.sport, fixture_id, sportmonks_key)
            else:
                logger.warning("Unknown external_id prefix %r for sport_prediction id=%d", prefix, row.id)
                skipped += 1
                continue
        except Exception as exc:
            logger.warning("Failed to fetch result for %s (id=%d): %s", row.external_id, row.id, exc)
            errors += 1
            continue

        if actual is None:
            logger.debug("No result yet for %s (sport=%s)", row.external_id, row.sport)
            skipped += 1
            continue

        if dry_run:
            logger.info("[dry-run] Would set %s → %s (predicted=%s)", row.external_id, actual, row.predicted_outcome)
            resolved += 1
            continue

        with get_session() as session:
            sp = session.get(SportPrediction, row.id)
            if sp is not None:
                sp.actual_outcome = actual
                sp.correct = sp.predicted_outcome == actual
        resolved += 1
        logger.info("Resolved %s: predicted=%s actual=%s correct=%s", row.external_id, row.predicted_outcome, actual, row.predicted_outcome == actual)

    logger.info("Sport prediction resolution complete: resolved=%d skipped=%d errors=%d", resolved, skipped, errors)
    return {"resolved": resolved, "skipped": skipped, "errors": errors}
