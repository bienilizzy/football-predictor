"""Tests for the batched weather-backfill helpers."""
from __future__ import annotations

import datetime as dt

from football_predictor.db.models import Match, Team, WeatherBackfillAttempt, WeatherRecord
from football_predictor.db.session import get_session
from football_predictor.ingestion.weather_backfill import (
    ARCHIVE_CUTOFF_DAYS,
    _needs_weather,
    _record_attempt,
    build_batches,
)


def test_build_batches_splits_into_even_chunks():
    batches = build_batches(list(range(100)), batch_size=50)

    assert len(batches) == 2
    assert batches[0] == list(range(0, 50))
    assert batches[1] == list(range(50, 100))


def test_build_batches_last_chunk_holds_the_remainder():
    batches = build_batches(list(range(110)), batch_size=50)

    assert len(batches) == 3
    assert [len(b) for b in batches] == [50, 50, 10]


def test_build_batches_empty_input():
    assert build_batches([], batch_size=50) == []


def test_build_batches_batch_size_larger_than_input():
    batches = build_batches([1, 2, 3], batch_size=50)

    assert batches == [[1, 2, 3]]


def test_needs_weather_when_no_record_exists():
    now = dt.datetime(2026, 1, 1)
    assert _needs_weather(None, kickoff_utc=now - dt.timedelta(days=30), now=now) is True


def test_needs_weather_false_for_non_forecast_record():
    now = dt.datetime(2026, 1, 1)
    wr = WeatherRecord(is_forecast=False)

    assert _needs_weather(wr, kickoff_utc=now - dt.timedelta(days=30), now=now) is False


def test_needs_weather_false_for_forecast_not_yet_stale():
    now = dt.datetime(2026, 1, 1)
    wr = WeatherRecord(is_forecast=True)
    kickoff = now - dt.timedelta(days=ARCHIVE_CUTOFF_DAYS - 1)

    assert _needs_weather(wr, kickoff_utc=kickoff, now=now) is False


def test_needs_weather_true_for_stale_forecast():
    now = dt.datetime(2026, 1, 1)
    wr = WeatherRecord(is_forecast=True)
    kickoff = now - dt.timedelta(days=ARCHIVE_CUTOFF_DAYS + 1)

    assert _needs_weather(wr, kickoff_utc=kickoff, now=now) is True


def test_record_attempt_increments_count_from_a_fresh_row():
    """Regression test: a brand-new WeatherBackfillAttempt has attempt_count=None
    in Python (the column default only applies on INSERT), so `+= 1` must not
    blow up on the first attempt for a match."""
    with get_session() as session:
        home, away = session.query(Team).order_by(Team.id).limit(2).all()
        match = Match(
            season="2099-test-attempt",
            home_team_id=home.id,
            away_team_id=away.id,
            kickoff_utc=dt.datetime(2026, 1, 1),
        )
        session.add(match)
        session.flush()
        match_id = match.id

    with get_session() as session:
        _record_attempt(session, match_id, success=True, error=None)

    with get_session() as session:
        attempt = session.get(WeatherBackfillAttempt, match_id)
        assert attempt.attempt_count == 1
        assert attempt.success is True
        assert attempt.last_error is None

    with get_session() as session:
        _record_attempt(session, match_id, success=False, error="boom")

    with get_session() as session:
        attempt = session.get(WeatherBackfillAttempt, match_id)
        assert attempt.attempt_count == 2
        assert attempt.success is False
        assert attempt.last_error == "boom"
