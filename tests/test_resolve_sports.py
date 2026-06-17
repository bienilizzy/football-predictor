"""Tests for the sport prediction resolution module.

Covers: fd_org result parsing, sportmonks result parsing per sport,
outcome-mapping logic, and the DB update path.
"""
from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch

import pytest

from football_predictor.db.models import SportPrediction
from football_predictor.db.session import get_session
from football_predictor.models.resolve_sports import (
    _parse_sportmonks_result,
    resolve_sport_predictions,
)


# --- Unit tests for _parse_sportmonks_result ---

def test_football_home_win():
    data = {
        "scores": [
            {"description": "CURRENT", "score": {"participant": "home", "goals": 2}},
            {"description": "CURRENT", "score": {"participant": "away", "goals": 0}},
        ]
    }
    assert _parse_sportmonks_result("football", data) == "H"


def test_football_away_win():
    data = {
        "scores": [
            {"description": "CURRENT", "score": {"participant": "home", "goals": 1}},
            {"description": "CURRENT", "score": {"participant": "away", "goals": 3}},
        ]
    }
    assert _parse_sportmonks_result("football", data) == "A"


def test_football_draw():
    data = {
        "scores": [
            {"description": "CURRENT", "score": {"participant": "home", "goals": 1}},
            {"description": "CURRENT", "score": {"participant": "away", "goals": 1}},
        ]
    }
    assert _parse_sportmonks_result("football", data) == "D"


def test_football_ignores_non_current_scores():
    data = {
        "scores": [
            {"description": "FIRST_HALF", "score": {"participant": "home", "goals": 1}},
            {"description": "FIRST_HALF", "score": {"participant": "away", "goals": 0}},
            {"description": "CURRENT", "score": {"participant": "home", "goals": 2}},
            {"description": "CURRENT", "score": {"participant": "away", "goals": 3}},
        ]
    }
    assert _parse_sportmonks_result("football", data) == "A"


def test_football_missing_scores_returns_none():
    assert _parse_sportmonks_result("football", {}) is None
    assert _parse_sportmonks_result("football", {"scores": []}) is None


def test_cricket_home_win_uses_runs():
    data = {
        "scores": [
            {"description": "CURRENT", "score": {"participant": "home", "runs": 280}},
            {"description": "CURRENT", "score": {"participant": "away", "runs": 220}},
        ]
    }
    assert _parse_sportmonks_result("cricket", data) == "H"


def test_tennis_home_player_wins():
    data = {
        "participants": [
            {"result": {"winner": True}},
            {"result": {"winner": False}},
        ]
    }
    assert _parse_sportmonks_result("tennis", data) == "H"


def test_tennis_away_player_wins():
    data = {
        "participants": [
            {"result": {"winner": False}},
            {"result": {"winner": True}},
        ]
    }
    assert _parse_sportmonks_result("tennis", data) == "A"


def test_tennis_no_winner_flag_returns_none():
    data = {"participants": [{"result": {}}, {"result": {}}]}
    assert _parse_sportmonks_result("tennis", data) is None


def test_f1_driver1_finishes_higher():
    data = {
        "participants": [
            {"result": {"position": 1}},
            {"result": {"position": 3}},
        ]
    }
    assert _parse_sportmonks_result("f1", data) == "H"


def test_f1_driver2_finishes_higher():
    data = {
        "participants": [
            {"result": {"position": 5}},
            {"result": {"position": 2}},
        ]
    }
    assert _parse_sportmonks_result("f1", data) == "A"


def test_f1_insufficient_participants_returns_none():
    data = {"participants": [{"result": {"position": 1}}]}
    assert _parse_sportmonks_result("f1", data) is None


# --- Integration tests for resolve_sport_predictions ---

def _seed_past_sport_prediction(sport: str, external_id: str, predicted_outcome: str) -> int:
    with get_session() as session:
        sp = SportPrediction(
            sport=sport,
            external_id=external_id,
            kickoff_utc=dt.datetime.now(dt.UTC).replace(tzinfo=None) - dt.timedelta(days=1),
            p_home=0.5,
            p_draw=0.3,
            p_away=0.2,
            predicted_outcome=predicted_outcome,
            confidence=0.5,
            source="llm_committee",
        )
        session.add(sp)
        session.flush()
        return sp.id


def test_resolve_updates_correct_outcome(monkeypatch):
    row_id = _seed_past_sport_prediction("football", "sportmonks:99001", "H")

    import football_predictor.models.resolve_sports as mod

    monkeypatch.setattr(mod, "_sportmonks_result", lambda sport, fid, key: "H")
    monkeypatch.setattr(mod.settings, "sportmonks_api_key", "fake-key")
    monkeypatch.setattr(mod.settings, "football_data_org_api_key", "")

    summary = resolve_sport_predictions()

    assert summary["resolved"] >= 1
    assert summary["errors"] == 0

    with get_session() as session:
        sp = session.get(SportPrediction, row_id)
        assert sp.actual_outcome == "H"
        assert sp.correct is True


def test_resolve_marks_incorrect_outcome(monkeypatch):
    row_id = _seed_past_sport_prediction("football", "sportmonks:99002", "H")

    import football_predictor.models.resolve_sports as mod

    monkeypatch.setattr(mod, "_sportmonks_result", lambda sport, fid, key: "A")
    monkeypatch.setattr(mod.settings, "sportmonks_api_key", "fake-key")
    monkeypatch.setattr(mod.settings, "football_data_org_api_key", "")

    resolve_sport_predictions()

    with get_session() as session:
        sp = session.get(SportPrediction, row_id)
        assert sp.actual_outcome == "A"
        assert sp.correct is False


def test_resolve_skips_when_api_key_missing(monkeypatch):
    row_id = _seed_past_sport_prediction("cricket", "sportmonks:99003", "H")

    import football_predictor.models.resolve_sports as mod

    monkeypatch.setattr(mod.settings, "sportmonks_api_key", "")
    monkeypatch.setattr(mod.settings, "football_data_org_api_key", "")

    summary = resolve_sport_predictions()

    assert summary["skipped"] >= 1
    assert summary["resolved"] == 0

    with get_session() as session:
        sp = session.get(SportPrediction, row_id)
        assert sp.actual_outcome is None


def test_resolve_dry_run_does_not_write(monkeypatch):
    row_id = _seed_past_sport_prediction("tennis", "sportmonks:99004", "A")

    import football_predictor.models.resolve_sports as mod

    monkeypatch.setattr(mod, "_sportmonks_result", lambda sport, fid, key: "A")
    monkeypatch.setattr(mod.settings, "sportmonks_api_key", "fake-key")
    monkeypatch.setattr(mod.settings, "football_data_org_api_key", "")

    summary = resolve_sport_predictions(dry_run=True)

    assert summary["resolved"] >= 1

    with get_session() as session:
        sp = session.get(SportPrediction, row_id)
        assert sp.actual_outcome is None


def test_resolve_skips_none_result(monkeypatch):
    row_id = _seed_past_sport_prediction("f1", "sportmonks:99005", "H")

    import football_predictor.models.resolve_sports as mod

    monkeypatch.setattr(mod, "_sportmonks_result", lambda sport, fid, key: None)
    monkeypatch.setattr(mod.settings, "sportmonks_api_key", "fake-key")
    monkeypatch.setattr(mod.settings, "football_data_org_api_key", "")

    summary = resolve_sport_predictions()

    assert summary["skipped"] >= 1
    assert summary["resolved"] == 0

    with get_session() as session:
        sp = session.get(SportPrediction, row_id)
        assert sp.actual_outcome is None
