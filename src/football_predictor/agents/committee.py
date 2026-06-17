"""Multi-agent LLM prediction committee.

`PredictionCommittee` runs five `SportAnalystAgent` personas in parallel via
`AsyncAnthropic`, each producing an independent {home_win, draw, away_win}
probability estimate for a fixture from a different analytical angle. If the
agents broadly agree (their per-class variance is below
`variance_threshold`), the averaged probabilities are returned directly as a
high-confidence "llm_committee" prediction. Otherwise `predict()` falls back
to the trained XGBoost model (see `football_predictor.models.predict`) for
football fixtures with stored features; if no such fallback is available
(other sports, or no active model), the committee average is returned anyway,
flagged as low-confidence.
"""
from __future__ import annotations

import asyncio
import logging

import anthropic
import numpy as np
import pandas as pd
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from config.settings import settings
from football_predictor.agents.cache import CommitteeResponseCache
from football_predictor.db.models import MatchFeatures
from football_predictor.db.session import get_session
from football_predictor.models.predict import load_active_model

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
DEFAULT_VARIANCE_THRESHOLD = 0.015

OUTCOME_KEYS = ("home_win", "draw", "away_win")


class AgentPrediction(BaseModel):
    """Structured response each analyst agent must return."""

    home_win: float = Field(ge=0.0, le=1.0)
    draw: float = Field(ge=0.0, le=1.0)
    away_win: float = Field(ge=0.0, le=1.0)
    reasoning: str


class SportAnalystAgent:
    """A single LLM persona that produces an independent outcome-probability estimate."""

    def __init__(self, name: str, expertise: str, system_prompt: str):
        self.name = name
        self.expertise = expertise
        self.system_prompt = system_prompt


def _fixture_summary(fixture: dict, sport: str) -> str:
    """Render a fixture dict as a compact text block for an agent prompt."""
    lines = [f"Sport: {sport}"]
    for key, value in fixture.items():
        if value is None:
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


class PredictionCommittee:
    """Orchestrates a committee of LLM analyst agents, with an XGBoost fallback."""

    def __init__(
        self,
        variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
        api_key: str | None = None,
        cache: CommitteeResponseCache | None = None,
        client: AsyncAnthropic | None = None,
    ):
        self.variance_threshold = variance_threshold
        if client is not None:
            self.client = client
        else:
            resolved_key = api_key if api_key is not None else (settings.anthropic_api_key or None)
            self.client = AsyncAnthropic(api_key=resolved_key)
        self.cache = cache if cache is not None else CommitteeResponseCache()
        self.agents = [
            SportAnalystAgent(
                name="Form Analyst",
                expertise="recent results, scoring trends, and momentum",
                system_prompt=(
                    "You are the Form Analyst on a sports prediction committee. "
                    "Focus exclusively on each side's recent form: results, "
                    "scoring/conceding trends, and momentum over their last "
                    "several matches. Estimate the probability of a home win, "
                    "draw, and away win based on form alone, and briefly explain "
                    "your reasoning."
                ),
            ),
            SportAnalystAgent(
                name="Tactical Analyst",
                expertise="playing styles, matchups, and tactical setups",
                system_prompt=(
                    "You are the Tactical Analyst on a sports prediction "
                    "committee. Focus on tactical matchups: playing styles, "
                    "formations, and strengths or weaknesses that favor one side "
                    "over the other. Estimate the probability of a home win, "
                    "draw, and away win based on the tactical picture, and "
                    "briefly explain your reasoning."
                ),
            ),
            SportAnalystAgent(
                name="Context Analyst",
                expertise="injuries, fixture congestion, motivation, and stakes",
                system_prompt=(
                    "You are the Context Analyst on a sports prediction "
                    "committee. Focus on situational factors: injuries or "
                    "suspensions, fixture congestion and rest, travel, "
                    "motivation, and what's at stake for each side. Estimate "
                    "the probability of a home win, draw, and away win based on "
                    "this context, and briefly explain your reasoning."
                ),
            ),
            SportAnalystAgent(
                name="Market Analyst",
                expertise="betting market signals and implied probabilities",
                system_prompt=(
                    "You are the Market Analyst on a sports prediction "
                    "committee. Focus on what betting markets and public "
                    "perception typically imply for a fixture like this, "
                    "adjusting for any mispricing you can infer from the data "
                    "given. Estimate the probability of a home win, draw, and "
                    "away win, and briefly explain your reasoning."
                ),
            ),
            SportAnalystAgent(
                name="Historical Pattern Analyst",
                expertise="head-to-head history and long-run patterns",
                system_prompt=(
                    "You are the Historical Pattern Analyst on a sports "
                    "prediction committee. Focus on head-to-head history and "
                    "long-run patterns between these two sides (and at this "
                    "venue, if relevant). Estimate the probability of a home "
                    "win, draw, and away win based on historical patterns, and "
                    "briefly explain your reasoning."
                ),
            ),
        ]

    async def get_agent_prediction(self, agent: SportAnalystAgent, fixture: dict, sport: str) -> AgentPrediction | None:
        """Ask one analyst agent for its independent outcome probabilities.

        Returns `None` (rather than raising) on any API error, so a single
        agent's failure doesn't take down the whole committee - `predict()`
        treats `None` as an abstention.
        """
        prompt = (
            f"{_fixture_summary(fixture, sport)}\n\n"
            "Give your independent probability estimate for this match's "
            "outcome from the home side's perspective. The three "
            "probabilities must sum to 1.0."
        )
        try:
            response = await self.client.messages.parse(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=agent.system_prompt,
                messages=[{"role": "user", "content": prompt}],
                output_format=AgentPrediction,
            )
        except (anthropic.APIError, TypeError) as exc:
            # A missing/invalid API key fails header validation with a bare
            # TypeError (not an AnthropicError subclass), before any request
            # is sent - treat it the same as an API error: this agent abstains.
            logger.warning("Committee agent %s failed: %s", agent.name, exc)
            return None

        return response.parsed_output

    async def _committee_opinions(self, fixture: dict, sport: str) -> list[tuple[SportAnalystAgent, AgentPrediction]]:
        """Run all agents in parallel, dropping any that failed."""
        results = await asyncio.gather(
            *(self.get_agent_prediction(agent, fixture, sport) for agent in self.agents)
        )
        return [(agent, result) for agent, result in zip(self.agents, results) if result is not None]

    def _xgboost_fallback(self, fixture: dict) -> dict | None:
        """Fall back to the trained XGBoost model for football fixtures with stored features.

        Returns `None` if `fixture` has no `match_id`, there's no active
        model, or no feature vector has been computed for that match -
        callers should treat that as "no fallback available".
        """
        match_id = fixture.get("match_id")
        if match_id is None:
            return None

        try:
            _, artifact = load_active_model()
        except RuntimeError:
            return None

        model = artifact["model"]
        feature_cols = artifact["feature_columns"]
        labels: list[str] = artifact["outcome_labels"]
        home_idx, draw_idx, away_idx = labels.index("H"), labels.index("D"), labels.index("A")

        with get_session() as session:
            mf = session.get(MatchFeatures, match_id)
            if mf is None:
                return None
            features = mf.features

        X = pd.DataFrame([{c: features.get(c, 0.0) for c in feature_cols}])
        p = model.predict_proba(X)[0]
        return {
            "source": "xgboost",
            "home_win": float(p[home_idx]),
            "draw": float(p[draw_idx]),
            "away_win": float(p[away_idx]),
        }

    async def committee_estimate(self, fixture: dict, sport: str = "football") -> dict | None:
        """Run all agents in parallel and return their aggregated estimate.

        Returns `None` if every agent failed. On success, returns a dict with
        `home_win`/`draw`/`away_win` (the agents' mean probabilities),
        `variance` (the largest per-class variance across agents), and
        `agent_opinions` (a per-agent breakdown). Exposed separately from
        `predict()` so callers (e.g. `models/ensemble_variance.py`) can
        combine the committee's estimate with their own fallback model.

        Results are cached (see `football_predictor.agents.cache`) for
        `CommitteeResponseCache.CACHE_TTL_SECONDS` (1 hour), keyed by `sport`
        and the fixture's `external_id` (or `match_id` for football's
        `/predictions/llm`) - a repeat request for the same fixture within
        that window skips the agent calls entirely.
        """
        fixture_id = fixture.get("external_id") or f"match:{fixture.get('match_id')}"

        cached = await self.cache.get(sport, fixture_id)
        if cached is not None:
            return cached

        opinions = await self._committee_opinions(fixture, sport)
        if not opinions:
            return None

        matrix = np.array(
            [[result.home_win, result.draw, result.away_win] for _, result in opinions], dtype=float
        )
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        mean_probs = matrix.mean(axis=0)
        max_variance = float(matrix.var(axis=0).max())

        agent_opinions = [
            {"agent": agent.name, **dict(zip(OUTCOME_KEYS, row.tolist())), "reasoning": result.reasoning}
            for (agent, result), row in zip(opinions, matrix)
        ]
        estimate = {
            **dict(zip(OUTCOME_KEYS, mean_probs.tolist())),
            "variance": max_variance,
            "agent_opinions": agent_opinions,
        }
        await self.cache.set(sport, fixture_id, estimate)
        return estimate

    async def predict(self, fixture: dict, sport: str = "football") -> dict:
        """Predict a fixture's outcome via the LLM committee, with an XGBoost fallback.

        Returns a dict with `source` ("llm_committee", "xgboost", or
        "llm_committee_low_confidence"), the averaged `home_win`/`draw`/
        `away_win` probabilities, and (when the committee ran) `variance` -
        the largest per-class variance across agents - and `agent_opinions`,
        a per-agent breakdown.

        If the committee's agents broadly agree (`variance` below
        `self.variance_threshold`), their averaged probabilities are returned
        directly. Otherwise this falls back to the trained XGBoost model for
        football fixtures with a `match_id` and stored features. If neither
        the committee nor the fallback is available, raises `RuntimeError`.
        """
        estimate = await self.committee_estimate(fixture, sport)

        if estimate is None:
            fallback = self._xgboost_fallback(fixture)
            if fallback is not None:
                return fallback
            raise RuntimeError("LLM committee unavailable and no fallback model available for this fixture")

        if estimate["variance"] < self.variance_threshold:
            return {"source": "llm_committee", **estimate}

        fallback = self._xgboost_fallback(fixture)
        if fallback is not None:
            fallback["variance"] = estimate["variance"]
            fallback["agent_opinions"] = estimate["agent_opinions"]
            return fallback

        return {"source": "llm_committee_low_confidence", **estimate}
