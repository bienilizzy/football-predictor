"""Pydantic response models for the API."""
from __future__ import annotations

import datetime as dt

from pydantic import BaseModel


class FixtureOut(BaseModel):
    match_id: int
    season: str
    kickoff_utc: dt.datetime
    home_team: str
    away_team: str
    status: str


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float


class PredictionOut(BaseModel):
    match_id: int
    kickoff_utc: dt.datetime
    home_team: str
    away_team: str
    predicted_outcome: str | None = None
    confidence: float | None = None
    p_home: float | None = None
    p_draw: float | None = None
    p_away: float | None = None
    top_features: list[FeatureContribution] | None = None


class AgentOpinionOut(BaseModel):
    agent: str
    home_win: float
    draw: float
    away_win: float
    reasoning: str


class AgentConsensusOut(BaseModel):
    agreeing: int
    total: int


class LlmPredictionOut(BaseModel):
    match_id: int
    kickoff_utc: dt.datetime
    home_team: str
    away_team: str
    source: str
    home_win: float
    draw: float
    away_win: float
    variance: float | None = None
    agent_opinions: list[AgentOpinionOut] | None = None


class OutcomeAccuracy(BaseModel):
    n_predicted: int
    accuracy: float


class AccuracySummaryOut(BaseModel):
    model_name: str
    n_predictions: int
    accuracy: float
    log_loss: float | None = None
    brier_score: float | None = None
    by_outcome: dict[str, OutcomeAccuracy] | None = None


class CalibrationCurveOut(BaseModel):
    model_name: str
    test_size: int
    curves: dict[str, dict[str, list[float]]]
    plot: dict | None = None


class AccuracyHistoryEntryOut(BaseModel):
    match_id: int
    kickoff_utc: dt.datetime
    home_team: str
    away_team: str
    predicted_outcome: str
    actual_outcome: str
    correct: bool
    p_home: float
    p_draw: float
    p_away: float


class TierAccuracyOut(BaseModel):
    tier: str
    min_confidence: float
    n_samples: int
    accuracy: float
    coverage_pct: float


class AccuracyByTierOut(BaseModel):
    model_name: str
    test_size: int
    tiers: dict[str, TierAccuracyOut]


class SportPredictionOut(BaseModel):
    sport: str
    external_id: str
    kickoff_utc: dt.datetime
    participants: dict[str, str | None]
    predicted_outcome: str | None = None
    confidence: float | None = None
    home_win: float | None = None
    draw: float | None = None
    away_win: float | None = None
    source: str | None = None
    consensus: AgentConsensusOut | None = None
    variance: float | None = None
    agent_opinions: list[AgentOpinionOut] | None = None


class SportLeaderboardEntryOut(BaseModel):
    sport: str
    n_resolved: int
    overall_accuracy: float
    tiers: dict[str, TierAccuracyOut]


class SportLeaderboardOut(BaseModel):
    sports: list[SportLeaderboardEntryOut]


class SportCalibrationOut(BaseModel):
    sport: str
    n_resolved: int
    curves: dict[str, dict[str, list[float]]]
    plot: dict | None = None


class SubscriptionTierOut(BaseModel):
    tier: str
    display_name: str
    monthly_price_usd: float
    description: str
    headline_accuracy: str
    limits: dict


class SubscriptionStatusOut(BaseModel):
    tier: str
    display_name: str
    monthly_price_usd: float
    description: str
    headline_accuracy: str
    owner_label: str
    daily_quota: int
    requests_today: int
    limits: dict
