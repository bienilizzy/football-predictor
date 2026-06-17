"""Tests for the LLM analyst committee (src/football_predictor/agents/committee.py).

Covers: mocked Anthropic responses (success and a single agent's failure), the
variance threshold that decides between the committee average and the XGBoost
fallback, per-sport prompt routing/caching, and the elite-only tier gate on
/predictions/llm (the free tier never reaches the committee).

All tests use a `FakeCache` in place of `CommitteeResponseCache` so they don't
require a running Redis instance.
"""
from __future__ import annotations

import asyncio
import datetime as dt
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

from football_predictor.agents.committee import AgentPrediction, PredictionCommittee, _fixture_summary
from football_predictor.api.main import app
from football_predictor.db.models import Match, Team
from football_predictor.db.session import get_session

client = TestClient(app)

FREE_KEY = "demo-free-key"
ELITE_KEY = "demo-elite-key"


def _auth(key: str) -> dict[str, str]:
    return {"X-API-Key": key}


class FakeCache:
    """In-memory stand-in for CommitteeResponseCache - keeps tests Redis-free."""

    def __init__(self):
        self.store: dict[tuple[str, str], dict] = {}

    async def get(self, sport: str, fixture_id: str) -> dict | None:
        return self.store.get((sport, fixture_id))

    async def set(self, sport: str, fixture_id: str, estimate: dict) -> None:
        self.store[(sport, fixture_id)] = estimate


def _new_committee(**kwargs) -> PredictionCommittee:
    """Build a PredictionCommittee with a FakeCache and a mock Anthropic client.

    Constructing a real AsyncAnthropic client is unrelated to what these tests
    exercise (and is slow in this environment), so a MagicMock stands in for it.
    """
    return PredictionCommittee(cache=FakeCache(), client=MagicMock(), **kwargs)


def _mock_agents(committee: PredictionCommittee, opinions: list[tuple[float, float, float]]) -> None:
    """Make get_agent_prediction return one canned opinion per agent, in order."""
    predictions = [
        AgentPrediction(home_win=h, draw=d, away_win=a, reasoning=f"{h}/{d}/{a}") for h, d, a in opinions
    ]

    async def fake(agent, fixture, sport):
        return predictions[committee.agents.index(agent)]

    committee.get_agent_prediction = fake


# Five agents agreeing closely -> variance well below the default 0.015 threshold.
LOW_VARIANCE_OPINIONS = [
    (0.60, 0.25, 0.15),
    (0.62, 0.24, 0.14),
    (0.58, 0.26, 0.16),
    (0.61, 0.23, 0.16),
    (0.59, 0.27, 0.14),
]

# Five agents in sharp disagreement -> variance well above the threshold.
HIGH_VARIANCE_OPINIONS = [
    (0.90, 0.05, 0.05),
    (0.10, 0.10, 0.80),
    (0.50, 0.40, 0.10),
    (0.30, 0.30, 0.40),
    (0.70, 0.20, 0.10),
]


# --- Mocking the Anthropic SDK ---

def test_get_agent_prediction_returns_parsed_response():
    committee = _new_committee()
    agent = committee.agents[0]
    expected = AgentPrediction(home_win=0.5, draw=0.3, away_win=0.2, reasoning="mocked response")

    committee.client.messages.parse = AsyncMock(return_value=SimpleNamespace(parsed_output=expected))

    fixture = {"external_id": "sportmonks:1", "home_team": "Arsenal", "away_team": "Chelsea"}
    result = asyncio.run(committee.get_agent_prediction(agent, fixture, "football"))

    assert result == expected
    _, kwargs = committee.client.messages.parse.call_args
    assert kwargs["output_format"] is AgentPrediction
    assert kwargs["system"] == agent.system_prompt


def test_get_agent_prediction_returns_none_on_api_error():
    committee = _new_committee()
    agent = committee.agents[0]

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    committee.client.messages.parse = AsyncMock(side_effect=anthropic.APIError("boom", request, body=None))

    fixture = {"external_id": "sportmonks:1", "home_team": "Arsenal", "away_team": "Chelsea"}
    result = asyncio.run(committee.get_agent_prediction(agent, fixture, "football"))

    assert result is None


# --- Variance threshold ---

def test_low_variance_returns_committee_average_directly():
    committee = _new_committee()
    _mock_agents(committee, LOW_VARIANCE_OPINIONS)

    fixture = {"external_id": "sportmonks:2", "home_team": "Arsenal", "away_team": "Chelsea"}
    result = asyncio.run(committee.predict(fixture, sport="football"))

    assert result["source"] == "llm_committee"
    assert result["variance"] < committee.variance_threshold
    assert result["home_win"] == pytest.approx(sum(o[0] for o in LOW_VARIANCE_OPINIONS) / 5)
    assert len(result["agent_opinions"]) == 5


def test_high_variance_without_fallback_is_low_confidence():
    committee = _new_committee()
    _mock_agents(committee, HIGH_VARIANCE_OPINIONS)

    # No match_id -> _xgboost_fallback returns None.
    fixture = {"external_id": "sportmonks:3", "home_team": "India", "away_team": "Australia"}
    result = asyncio.run(committee.predict(fixture, sport="cricket"))

    assert result["source"] == "llm_committee_low_confidence"
    assert result["variance"] >= committee.variance_threshold


# --- XGBoost fallback ---

def test_high_variance_falls_back_to_xgboost(monkeypatch):
    committee = _new_committee()
    _mock_agents(committee, HIGH_VARIANCE_OPINIONS)

    monkeypatch.setattr(
        committee,
        "_xgboost_fallback",
        lambda fixture: {"source": "xgboost", "home_win": 0.7, "draw": 0.2, "away_win": 0.1},
    )

    fixture = {"match_id": 123, "home_team": "Arsenal", "away_team": "Chelsea"}
    result = asyncio.run(committee.predict(fixture, sport="football"))

    assert result["source"] == "xgboost"
    assert result["home_win"] == 0.7
    # The committee's disagreement is still surfaced alongside the fallback prediction.
    assert result["variance"] >= committee.variance_threshold
    assert len(result["agent_opinions"]) == 5


def test_xgboost_fallback_returns_none_without_match_id():
    committee = _new_committee()
    assert committee._xgboost_fallback({"external_id": "sportmonks:4"}) is None


# --- Per-sport routing ---

@pytest.mark.parametrize("sport", ["football", "cricket", "tennis", "f1"])
def test_fixture_summary_includes_sport(sport):
    fixture = {"home_team": "A", "away_team": "B"}
    assert _fixture_summary(fixture, sport).startswith(f"Sport: {sport}")


def test_fixture_summary_handles_f1_driver_fields():
    fixture = {"driver1": "Verstappen", "driver2": "Hamilton"}
    summary = _fixture_summary(fixture, "f1")

    assert summary.startswith("Sport: f1")
    assert "driver1: Verstappen" in summary
    assert "driver2: Hamilton" in summary


def test_committee_estimate_passes_sport_to_each_agent():
    committee = _new_committee()
    seen_sports = []

    async def fake(agent, fixture, sport):
        seen_sports.append(sport)
        return AgentPrediction(home_win=0.5, draw=0.3, away_win=0.2, reasoning="ok")

    committee.get_agent_prediction = fake

    fixture = {"external_id": "sportmonks:5", "driver1": "Verstappen", "driver2": "Hamilton"}
    asyncio.run(committee.committee_estimate(fixture, sport="f1"))

    assert seen_sports == ["f1"] * 5


def test_committee_estimate_caches_per_sport_for_same_fixture():
    committee = _new_committee()
    call_count = {"n": 0}

    async def fake(agent, fixture, sport):
        call_count["n"] += 1
        return AgentPrediction(home_win=0.5, draw=0.3, away_win=0.2, reasoning="ok")

    committee.get_agent_prediction = fake
    fixture = {"external_id": "sportmonks:6", "home_team": "A", "away_team": "B"}

    asyncio.run(committee.committee_estimate(fixture, sport="football"))
    assert call_count["n"] == 5

    # Same fixture + sport -> cache hit, agents are not re-queried.
    asyncio.run(committee.committee_estimate(fixture, sport="football"))
    assert call_count["n"] == 5

    # Same fixture, different sport -> separate cache entry, agents run again.
    asyncio.run(committee.committee_estimate(fixture, sport="cricket"))
    assert call_count["n"] == 10


# --- Tier gating on /predictions/llm ---

def _seed_match(season: str) -> int:
    with get_session() as session:
        home, away = session.query(Team).order_by(Team.id).limit(2).all()
        match = Match(
            season=season,
            kickoff_utc=dt.datetime.now(dt.UTC).replace(tzinfo=None) + dt.timedelta(days=1),
            home_team_id=home.id,
            away_team_id=away.id,
            status="SCHEDULED",
        )
        session.add(match)
        session.flush()
        return match.id


def test_free_tier_llm_prediction_is_forbidden_without_calling_the_committee(monkeypatch):
    from football_predictor.api.routers import predictions as predictions_router

    match_id = _seed_match("2099-test-llm-free")

    async def fail_if_called(agent, fixture, sport):
        raise AssertionError("free tier must not reach the LLM committee")

    monkeypatch.setattr(predictions_router._committee, "get_agent_prediction", fail_if_called)

    resp = client.post("/api/v1/predictions/llm", json={"match_id": match_id}, headers=_auth(FREE_KEY))

    assert resp.status_code == 403


def test_elite_tier_llm_prediction_uses_the_committee(monkeypatch):
    from football_predictor.api.routers import predictions as predictions_router

    match_id = _seed_match("2099-test-llm-elite")

    monkeypatch.setattr(predictions_router._committee, "cache", FakeCache())
    _mock_agents(predictions_router._committee, LOW_VARIANCE_OPINIONS)

    resp = client.post("/api/v1/predictions/llm", json={"match_id": match_id}, headers=_auth(ELITE_KEY))

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "llm_committee"
    assert len(body["agent_opinions"]) == 5
