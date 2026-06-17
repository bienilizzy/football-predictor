"""Batched, resumable, rate-limit-aware weather backfill for `WeatherRecord`.

The full set of match ids is split into fixed-size batches and frozen into a
`WeatherBackfillCheckpoint` the first time a campaign starts, so resumed runs
keep the same batch boundaries no matter how much progress has been made.
Between batches the run sleeps `BATCH_SLEEP_SECONDS` to stay well under
Open-Meteo's connection-rate limits (see `WinError 10065` / DNS-rate-limit
issues that `weather_client._cached_dns_for_open_meteo` already works around).

Each match is attempted at most `MAX_RETRIES` times with exponential backoff
on HTTP 429 (Too Many Requests); the outcome and timestamp of the most recent
attempt is recorded in `WeatherBackfillAttempt` so failures can be inspected
and retried on a later run without redoing matches that already succeeded.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
import urllib.error

from football_predictor.db.models import (
    Match,
    Team,
    WeatherBackfillAttempt,
    WeatherBackfillCheckpoint,
    WeatherRecord,
)
from football_predictor.db.session import get_session
from football_predictor.ingestion.weather_client import (
    ARCHIVE_CUTOFF_DAYS,
    REQUEST_DELAY_SECONDS,
    _cached_dns_for_open_meteo,
    fetch_weather,
)

logger = logging.getLogger(__name__)

CHECKPOINT_ID = 1
BATCH_SIZE = 50
BATCH_SLEEP_SECONDS = 10.0
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0


def build_batches(match_ids: list[int], batch_size: int) -> list[list[int]]:
    """Split `match_ids` into consecutive chunks of at most `batch_size`."""
    return [match_ids[i : i + batch_size] for i in range(0, len(match_ids), batch_size)]


def _load_or_create_checkpoint(session, batch_size: int) -> WeatherBackfillCheckpoint:
    """Return the active checkpoint, (re)starting a campaign if needed.

    An existing checkpoint is reused as-is unless its batch plan is fully
    completed AND the set of matches in the DB has grown since it was made
    (e.g. a new fixture was ingested) - in that case a fresh checkpoint
    covering the current full set of matches is created.
    """
    all_match_ids = [row[0] for row in session.query(Match.id).order_by(Match.id).all()]
    checkpoint = session.get(WeatherBackfillCheckpoint, CHECKPOINT_ID)

    if checkpoint is not None:
        batches = build_batches(checkpoint.match_ids, checkpoint.batch_size)
        campaign_finished = checkpoint.last_completed_batch >= len(batches) - 1
        new_matches = set(all_match_ids) - set(checkpoint.match_ids)
        if not campaign_finished or not new_matches:
            return checkpoint

    if checkpoint is None:
        checkpoint = WeatherBackfillCheckpoint(id=CHECKPOINT_ID)
        session.add(checkpoint)

    checkpoint.match_ids = all_match_ids
    checkpoint.batch_size = batch_size
    checkpoint.last_completed_batch = -1
    checkpoint.updated_at = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    return checkpoint


def _record_attempt(session, match_id: int, success: bool, error: str | None) -> None:
    attempt = session.get(WeatherBackfillAttempt, match_id)
    if attempt is None:
        attempt = WeatherBackfillAttempt(match_id=match_id, attempt_count=0)
        session.add(attempt)
    attempt.attempt_count += 1
    attempt.last_attempted_at = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    attempt.success = success
    attempt.last_error = error


def _needs_weather(wr: WeatherRecord | None, kickoff_utc: dt.datetime, now: dt.datetime) -> bool:
    if wr is None:
        return True
    return wr.is_forecast and kickoff_utc < now - dt.timedelta(days=ARCHIVE_CUTOFF_DAYS)


def _process_match(session, match: Match, home_team: Team, now: dt.datetime) -> str:
    """Ensure `match` has up-to-date weather. Returns 'skipped'/'success'/'failed'."""
    wr = session.query(WeatherRecord).filter_by(match_id=match.id).one_or_none()
    if not _needs_weather(wr, match.kickoff_utc, now):
        return "skipped"

    backoff = INITIAL_BACKOFF_SECONDS
    for attempt_n in range(1, MAX_RETRIES + 1):
        try:
            result = fetch_weather(home_team.latitude, home_team.longitude, match.kickoff_utc)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt_n < MAX_RETRIES:
                logger.warning(
                    "429 from Open-Meteo for match %d (attempt %d/%d), backing off %.1fs",
                    match.id, attempt_n, MAX_RETRIES, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                continue
            _record_attempt(session, match.id, success=False, error=str(exc))
            return "failed"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.exception("Weather lookup failed for match %d", match.id)
            _record_attempt(session, match.id, success=False, error=str(exc))
            return "failed"

        if result is None:
            _record_attempt(session, match.id, success=False, error="no hourly data returned")
            return "failed"

        if wr is None:
            wr = WeatherRecord(match_id=match.id)
            session.add(wr)
        wr.temp_c = result["temp_c"]
        wr.precip_mm = result["precip_mm"]
        wr.wind_kph = result["wind_kph"]
        wr.is_forecast = result["is_forecast"]

        _record_attempt(session, match.id, success=True, error=None)
        return "success"

    return "failed"


def run_batched_backfill(
    batch_size: int = BATCH_SIZE, batch_sleep_seconds: float = BATCH_SLEEP_SECONDS
) -> dict:
    """Run (or resume) the batched weather backfill campaign.

    Returns a summary dict with the batch plan and per-match outcome counts
    for this run.
    """
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)

    with get_session() as session:
        checkpoint = _load_or_create_checkpoint(session, batch_size)
        match_ids = list(checkpoint.match_ids)
        effective_batch_size = checkpoint.batch_size
        start_batch = checkpoint.last_completed_batch + 1
        teams_by_id = {t.id: t for t in session.query(Team).all()}

    batches = build_batches(match_ids, effective_batch_size)
    n_batches = len(batches)

    counts = {"skipped": 0, "success": 0, "failed": 0}

    if start_batch >= n_batches:
        logger.info("Weather backfill already complete: %d/%d batches done.", n_batches, n_batches)
        return {
            "total_batches": n_batches,
            "batches_run": 0,
            "last_completed_batch": checkpoint.last_completed_batch if n_batches else -1,
            **counts,
        }

    logger.info(
        "Weather backfill: %d matches in %d batches of %d, resuming at batch %d/%d",
        len(match_ids), n_batches, effective_batch_size, start_batch + 1, n_batches,
    )

    last_completed_batch = start_batch - 1

    with _cached_dns_for_open_meteo():
        for batch_index in range(start_batch, n_batches):
            batch = batches[batch_index]
            logger.info("Batch %d/%d: %d matches", batch_index + 1, n_batches, len(batch))

            for match_id in batch:
                with get_session() as session:
                    match = session.get(Match, match_id)
                    home_team = teams_by_id.get(match.home_team_id)
                    if home_team is None:
                        _record_attempt(session, match_id, success=False, error="home team not found")
                        outcome = "failed"
                    else:
                        outcome = _process_match(session, match, home_team, now)

                counts[outcome] += 1
                if outcome != "skipped":
                    time.sleep(REQUEST_DELAY_SECONDS)

            last_completed_batch = batch_index
            with get_session() as session:
                cp = session.get(WeatherBackfillCheckpoint, CHECKPOINT_ID)
                cp.last_completed_batch = batch_index
                cp.updated_at = dt.datetime.now(dt.UTC).replace(tzinfo=None)

            if batch_index < n_batches - 1:
                logger.info(
                    "Batch %d/%d done; sleeping %.0fs before next batch",
                    batch_index + 1, n_batches, batch_sleep_seconds,
                )
                time.sleep(batch_sleep_seconds)

    return {
        "total_batches": n_batches,
        "batches_run": last_completed_batch - start_batch + 1,
        "last_completed_batch": last_completed_batch,
        **counts,
    }
