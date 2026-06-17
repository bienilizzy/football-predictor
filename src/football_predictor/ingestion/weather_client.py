"""Kickoff weather via the free, no-API-key Open-Meteo APIs.

- Historical: archive-api.open-meteo.com (has a short ~5 day publication delay)
- Forecast / recent: api.open-meteo.com (covers ~92 days back to 16 days ahead)
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

from football_predictor.db.models import Match, Team, WeatherRecord
from football_predictor.db.session import get_session

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = "temperature_2m,precipitation,windspeed_10m"

# Open-Meteo's historical archive lags a few days behind real time; for kickoffs
# more recent than this, fall back to the forecast API (which also covers recent
# past via start_date/end_date).
ARCHIVE_CUTOFF_DAYS = 5
REQUEST_DELAY_SECONDS = 0.2

_DNS_CACHE_HOSTS = {"archive-api.open-meteo.com", "api.open-meteo.com"}
_DNS_CACHE_TTL_SECONDS = 300


@contextlib.contextmanager
def _cached_dns_for_open_meteo():
    """Cache `socket.getaddrinfo` results for the Open-Meteo hosts.

    Their DNS records have a ~6s TTL; re-resolving on every one of ~1000
    sequential requests in a bulk backfill eventually trips DNS-level rate
    limiting and starts failing with `getaddrinfo failed`. Caching the
    resolution for the duration of the run avoids that.
    """
    cache: dict[tuple, tuple[float, object]] = {}
    original = socket.getaddrinfo

    def patched(host, *args, **kwargs):
        if host not in _DNS_CACHE_HOSTS:
            return original(host, *args, **kwargs)
        key = (host, args, tuple(sorted(kwargs.items())))
        now = time.monotonic()
        hit = cache.get(key)
        if hit and now - hit[0] < _DNS_CACHE_TTL_SECONDS:
            return hit[1]
        result = original(host, *args, **kwargs)
        cache[key] = (now, result)
        return result

    socket.getaddrinfo = patched
    try:
        yield
    finally:
        socket.getaddrinfo = original


def fetch_weather(lat: float, lon: float, kickoff_utc: dt.datetime) -> dict | None:
    """Return {temp_c, precip_mm, wind_kph, is_forecast} for the hour of kickoff."""
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    date_str = kickoff_utc.strftime("%Y-%m-%d")
    use_forecast_api = kickoff_utc > now - dt.timedelta(days=ARCHIVE_CUTOFF_DAYS)

    base_url = FORECAST_URL if use_forecast_api else ARCHIVE_URL
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "timezone": "UTC",
        "start_date": date_str,
        "end_date": date_str,
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    # Uses urllib (stdlib) rather than requests/httpx: both of those libraries hang
    # indefinitely (ignoring their `timeout` kwarg) against Open-Meteo's hosts on
    # some Windows/network setups, while urllib.request.urlopen works fine.
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return None

    target_hour = kickoff_utc.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M")
    try:
        idx = times.index(target_hour)
    except ValueError:
        idx = len(times) // 2  # fall back to a midday reading

    return {
        "temp_c": hourly["temperature_2m"][idx],
        "precip_mm": hourly["precipitation"][idx],
        "wind_kph": hourly["windspeed_10m"][idx],
        "is_forecast": kickoff_utc > now,
    }


def ingest_weather(limit: int | None = None) -> int:
    """Backfill/refresh WeatherRecord rows for matches that need it.

    A match "needs" weather if it has none yet, or if its only record is a
    forecast and the match is now old enough that the (more accurate) historical
    archive should have data.
    """
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    processed = 0

    with get_session() as session, _cached_dns_for_open_meteo():
        matches = session.query(Match).all()
        existing = {w.match_id: w for w in session.query(WeatherRecord).all()}
        teams_by_id = {t.id: t for t in session.query(Team).all()}

        for match in matches:
            wr = existing.get(match.id)
            stale_forecast = wr is not None and wr.is_forecast and match.kickoff_utc < now - dt.timedelta(
                days=ARCHIVE_CUTOFF_DAYS
            )
            if wr is not None and not stale_forecast:
                continue

            home_team = teams_by_id.get(match.home_team_id)
            if home_team is None:
                continue

            try:
                result = fetch_weather(home_team.latitude, home_team.longitude, match.kickoff_utc)
            except (urllib.error.URLError, OSError, ValueError):
                logger.exception("Weather lookup failed for match %s", match.id)
                continue
            if result is None:
                continue

            if wr is None:
                wr = WeatherRecord(match_id=match.id)
                session.add(wr)
            wr.temp_c = result["temp_c"]
            wr.precip_mm = result["precip_mm"]
            wr.wind_kph = result["wind_kph"]
            wr.is_forecast = result["is_forecast"]

            processed += 1
            if limit and processed >= limit:
                break
            time.sleep(REQUEST_DELAY_SECONDS)

    return processed
