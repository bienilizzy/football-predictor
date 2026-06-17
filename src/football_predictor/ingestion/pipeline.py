"""Top-level ingestion orchestration.

`ingest_historical` backfills past seasons (results, referees, cards, shots, xG,
weather) to build a training dataset. `ingest_latest` is meant to run on a daily
schedule: refresh the current season's results/xG, pull the upcoming fixture
schedule, and fetch weather (forecast for upcoming, archive for recently played).
"""
from __future__ import annotations

import logging

import requests

from config.settings import settings
from football_predictor.ingestion import football_data_co_uk, football_data_org, understat_client, weather_client

logger = logging.getLogger(__name__)


def ingest_historical(seasons: list[str] | None = None, with_weather: bool = True) -> dict:
    seasons = seasons or settings.historical_season_codes
    summary: dict = {"seasons": {}}

    for season_code in seasons:
        n_matches = football_data_co_uk.ingest_season(season_code)
        try:
            n_xg = understat_client.ingest_season_xg(season_code)
        except Exception:
            logger.exception("Understat xG ingestion failed for season %s", season_code)
            n_xg = 0
        summary["seasons"][season_code] = {"matches": n_matches, "xg_matches": n_xg}
        logger.info("Season %s: %d matches, %d xG rows", season_code, n_matches, n_xg)

    if with_weather:
        n_weather = weather_client.ingest_weather()
        summary["weather"] = n_weather
        logger.info("Weather: %d records", n_weather)

    return summary


def ingest_latest(
    days_ahead: int = 14, days_back: int = 7, with_weather: bool = True, force_recent: bool = False
) -> dict:
    """Refresh fixtures/results/weather for the current season.

    If `force_recent` is True, skips the football-data.co.uk/Understat
    historical-style backfill of the current season and goes straight to the
    football-data.org fixtures/results refresh - useful for a quick "just get
    what's new" run, e.g. during the off-season when there's no current-season
    CSV yet.
    """
    summary: dict = {}
    current_code = football_data_co_uk.label_to_season_code(settings.current_season)

    if not force_recent:
        try:
            summary["current_season_matches"] = football_data_co_uk.ingest_season(current_code, force_download=True)
        except requests.HTTPError:
            logger.warning("football-data.co.uk has no file yet for current season %s", current_code)
            summary["current_season_matches"] = 0

        try:
            summary["current_season_xg"] = understat_client.ingest_season_xg(current_code)
        except Exception:
            logger.exception("Understat xG ingestion failed for current season %s", current_code)
            summary["current_season_xg"] = 0

    summary["upcoming_fixtures"] = football_data_org.ingest_upcoming_fixtures(days_ahead=days_ahead)
    summary["recent_results"] = football_data_org.ingest_recent_results(days_back=days_back)

    if summary["upcoming_fixtures"] == 0:
        logger.info("No upcoming fixtures for %s; checking fallback leagues", settings.fd_org_competition_code)
        summary["fallback_fixtures"] = football_data_org.ingest_fallback_fixtures(days_ahead=days_ahead)

    if with_weather:
        summary["weather"] = weather_client.ingest_weather()

    return summary
