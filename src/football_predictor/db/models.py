"""SQLAlchemy ORM models for the football prediction system."""
from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    fd_org_name: Mapped[str] = mapped_column(String(64))
    fd_co_uk_name: Mapped[str] = mapped_column(String(64))
    understat_name: Mapped[str] = mapped_column(String(64))
    stadium: Mapped[str] = mapped_column(String(128))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("season", "home_team_id", "away_team_id", name="uq_match_fixture"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    season: Mapped[str] = mapped_column(String(16), index=True)
    matchday: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    kickoff_utc: Mapped[dt.datetime] = mapped_column(DateTime, index=True)

    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)

    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(String(16), default="SCHEDULED")  # SCHEDULED | FINISHED

    referee_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    home_yellow: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_yellow: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_red: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_red: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_fouls: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_fouls: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    fd_org_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, unique=True)

    home_team: Mapped["Team"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship(foreign_keys=[away_team_id])

    @property
    def result(self) -> Optional[str]:
        """H / D / A based on final score, or None if not finished."""
        if self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return "H"
        if self.home_score < self.away_score:
            return "A"
        return "D"


class OtherLeagueFixture(Base):
    """Fixture from a non-Premier-League competition.

    Used as a fallback "what's on" source when the Premier League has no
    upcoming fixtures (e.g. its summer off-season). Stored independently of
    `Match`/`Team`, which are scoped to the PL's 20-team roster - these rows
    carry team names as plain strings and aren't fed into the prediction
    pipeline.
    """

    __tablename__ = "other_league_fixtures"

    id: Mapped[int] = mapped_column(primary_key=True)
    fd_org_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    competition_code: Mapped[str] = mapped_column(String(16))
    competition_name: Mapped[str] = mapped_column(String(64))
    season: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)

    home_team_name: Mapped[str] = mapped_column(String(64))
    away_team_name: Mapped[str] = mapped_column(String(64))
    kickoff_utc: Mapped[dt.datetime] = mapped_column(DateTime, index=True)
    status: Mapped[str] = mapped_column(String(16), default="SCHEDULED")

    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class MatchStats(Base):
    """Per-team Understat advanced stats for a match (xG, shots, PPDA, ...)."""

    __tablename__ = "match_stats"
    __table_args__ = (UniqueConstraint("match_id", "team_id", name="uq_match_team_stats"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    is_home: Mapped[bool] = mapped_column(Boolean)

    xg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    xga: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    shots: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shots_on_target: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    deep_completions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ppda: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class WeatherRecord(Base):
    """Weather at kickoff for a match (historical observation or forecast)."""

    __tablename__ = "weather_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), unique=True, index=True)

    temp_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precip_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_kph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_forecast: Mapped[bool] = mapped_column(Boolean, default=False)


class MatchFeatures(Base):
    """Wide engineered feature row for a match, ready for model train/predict."""

    __tablename__ = "match_features"

    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), primary_key=True)
    features: Mapped[dict] = mapped_column(JSON)
    target: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)  # H | D | A
    computed_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    trained_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)
    feature_names: Mapped[list] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)
    artifact_path: Mapped[str] = mapped_column(String(256))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("match_id", "model_version_id", name="uq_match_model_pred"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))

    p_home: Mapped[float] = mapped_column(Float)
    p_draw: Mapped[float] = mapped_column(Float)
    p_away: Mapped[float] = mapped_column(Float)
    predicted_outcome: Mapped[str] = mapped_column(String(1))
    confidence: Mapped[float] = mapped_column(Float)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


class RejectedPrediction(Base):
    """Predictions whose top-class probability fell below the confidence threshold.

    Kept separately from `Prediction` (which only holds predictions the model
    was confident enough to surface) so low-confidence cases can be reviewed
    later without polluting accuracy tracking for surfaced predictions.
    """

    __tablename__ = "rejected_predictions"
    __table_args__ = (
        UniqueConstraint("match_id", "model_version_id", name="uq_match_model_rejected"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))

    p_home: Mapped[float] = mapped_column(Float)
    p_draw: Mapped[float] = mapped_column(Float)
    p_away: Mapped[float] = mapped_column(Float)
    predicted_outcome: Mapped[str] = mapped_column(String(1))
    confidence: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


class PredictionResult(Base):
    """Scoring of a prediction once the match has been played."""

    __tablename__ = "prediction_results"

    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), primary_key=True)
    actual_outcome: Mapped[str] = mapped_column(String(1))
    correct: Mapped[bool] = mapped_column(Boolean)
    brier_score: Mapped[float] = mapped_column(Float)
    log_loss: Mapped[float] = mapped_column(Float)
    scored_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


class SportPrediction(Base):
    """A multi-sport prediction (LLM analyst committee + XGBoost), tracked for accuracy.

    Written by `GET /sports/{sport}/upcoming` for football, cricket, tennis,
    and f1 fixtures alike (unlike `Prediction`, which is football/PL-only).
    `actual_outcome`/`correct` stay `None` until a separate resolution step
    fills them in once the fixture is played; `GET /sports/leaderboard`
    aggregates over the resolved rows.
    """

    __tablename__ = "sport_predictions"
    __table_args__ = (
        UniqueConstraint("sport", "external_id", name="uq_sport_prediction_fixture"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    sport: Mapped[str] = mapped_column(String(16), index=True)
    external_id: Mapped[str] = mapped_column(String(64), index=True)
    kickoff_utc: Mapped[dt.datetime] = mapped_column(DateTime, index=True)

    p_home: Mapped[float] = mapped_column(Float)
    p_draw: Mapped[float] = mapped_column(Float)
    p_away: Mapped[float] = mapped_column(Float)
    predicted_outcome: Mapped[str] = mapped_column(String(1))
    confidence: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32))

    actual_outcome: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)
    correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )


class WeatherBackfillCheckpoint(Base):
    """Frozen batch plan + progress marker for a batched weather-backfill run.

    Singleton row (id=1). `match_ids` is the ordered list of match ids the
    campaign is working through, fixed when the campaign starts so batch
    boundaries stay stable across resumed runs even as matches drop out of
    the "needs weather" pool.
    """

    __tablename__ = "weather_backfill_checkpoint"

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    match_ids: Mapped[list] = mapped_column(JSON)
    batch_size: Mapped[int] = mapped_column(Integer)
    last_completed_batch: Mapped[int] = mapped_column(Integer, default=-1)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)


class WeatherBackfillAttempt(Base):
    """Per-match retry log for the batched weather backfill."""

    __tablename__ = "weather_backfill_attempts"

    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), primary_key=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    last_attempted_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    last_error: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)


class ApiKey(Base):
    """Subscription-tier API key for accessing the prediction API."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(primary_key=True)
    key_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    owner_label: Mapped[str] = mapped_column(String(64))
    tier: Mapped[str] = mapped_column(String(16))  # free | pro | elite

    daily_quota: Mapped[int] = mapped_column(Integer)
    requests_today: Mapped[int] = mapped_column(Integer, default=0)
    quota_reset_at: Mapped[dt.datetime] = mapped_column(DateTime)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)
