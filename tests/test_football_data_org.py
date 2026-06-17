"""Tests for fallback-league competition lookup and ingestion."""
from __future__ import annotations

import datetime as dt

from football_predictor.db.models import OtherLeagueFixture
from football_predictor.db.session import get_session
from football_predictor.ingestion import football_data_org
from football_predictor.ingestion.football_data_org import _find_competition_code

COMPETITIONS = [
    {"code": "PL", "name": "Premier League"},
    {"code": "ASL", "name": "Allsvenskan"},
    {"code": "ELS", "name": "Norwegian Eliteserien"},
]


def test_finds_competition_by_exact_name():
    assert _find_competition_code(COMPETITIONS, "Allsvenskan") == "ASL"


def test_match_is_case_insensitive():
    assert _find_competition_code(COMPETITIONS, "allsvenskan") == "ASL"


def test_match_is_substring_of_competition_name():
    assert _find_competition_code(COMPETITIONS, "Eliteserien") == "ELS"


def test_returns_none_when_not_found():
    assert _find_competition_code(COMPETITIONS, "J1 League") is None


def test_returns_first_match():
    competitions = [
        {"code": "FOO", "name": "Foo Eliteserien Bar"},
        {"code": "ELS", "name": "Eliteserien"},
    ]
    assert _find_competition_code(competitions, "Eliteserien") == "FOO"


def _fixture_payload(fd_id: int, home: str, away: str) -> dict:
    kickoff = dt.datetime.now(dt.UTC).replace(tzinfo=None) + dt.timedelta(days=1)
    return {
        "id": fd_id,
        "utcDate": kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "SCHEDULED",
        "season": {"startDate": "2026-03-01"},
        "homeTeam": {"name": home},
        "awayTeam": {"name": away},
        "score": {"fullTime": {"home": None, "away": None}},
    }


def test_ingest_fallback_fixtures_stores_matched_leagues(monkeypatch):
    """Allsvenskan/Eliteserien are on the (mocked) plan, J1 League is not."""
    monkeypatch.setattr(football_data_org.settings, "football_data_org_api_key", "dummy-key")

    competitions = [
        {"code": "PL", "name": "Premier League"},
        {"code": "ASL", "name": "Allsvenskan"},
        {"code": "ELS", "name": "Eliteserien"},
    ]
    fixtures_by_code = {
        "ASL": [_fixture_payload(900_001, "AIK", "Hammarby")],
        "ELS": [_fixture_payload(900_002, "Bodo/Glimt", "Molde")],
    }

    monkeypatch.setattr(football_data_org.FootballDataOrgClient, "fetch_competitions", lambda self: competitions)
    monkeypatch.setattr(
        football_data_org.FootballDataOrgClient,
        "fetch_upcoming_fixtures",
        lambda self, days_ahead=14, competition_code=None: fixtures_by_code.get(competition_code, []),
    )

    results = football_data_org.ingest_fallback_fixtures(days_ahead=14)

    assert results == {"Allsvenskan": 1, "Eliteserien": 1}

    with get_session() as session:
        by_id = {
            f.fd_org_id: f
            for f in session.query(OtherLeagueFixture)
            .filter(OtherLeagueFixture.fd_org_id.in_([900_001, 900_002]))
            .all()
        }
        assert by_id[900_001].competition_code == "ASL"
        assert by_id[900_001].home_team_name == "AIK"
        assert by_id[900_002].competition_code == "ELS"
        assert by_id[900_002].away_team_name == "Molde"


def test_ingest_fallback_fixtures_handles_no_leagues_available(monkeypatch):
    """If none of the fallback leagues are on the API plan, returns {} without error."""
    monkeypatch.setattr(football_data_org.settings, "football_data_org_api_key", "dummy-key")
    monkeypatch.setattr(
        football_data_org.FootballDataOrgClient,
        "fetch_competitions",
        lambda self: [{"code": "PL", "name": "Premier League"}],
    )

    assert football_data_org.ingest_fallback_fixtures(days_ahead=14) == {}
