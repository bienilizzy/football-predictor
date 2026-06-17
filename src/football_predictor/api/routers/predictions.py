"""Prediction endpoints, gated by tier (probabilities, horizon, feature contributions)."""
from __future__ import annotations
from football_predictor.bookmakers.bet9ja import Bet9jaAdapter
from football_predictor.bookmakers.onexbet import OnexBetAdapter
import datetime as dt
import logging
from collections import defaultdict

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from football_predictor.agents.committee import OUTCOME_KEYS, PredictionCommittee
from football_predictor.api.auth import AuthContext, get_auth_context
from football_predictor.api.schemas import (
    AgentConsensusOut,
    AgentOpinionOut,
    FeatureContribution,
    LlmPredictionOut,
    PredictionOut,
    SportCalibrationOut,
    SportLeaderboardEntryOut,
    SportLeaderboardOut,
    SportPredictionOut,
    TierAccuracyOut,
)
from football_predictor.db.models import Match, ModelVersion, Prediction, SportPrediction, Team
from football_predictor.db.session import get_session
from football_predictor.models.evaluation import calibration_curve_data, calibration_curve_figure
from football_predictor.models.predict import DEFAULT_CONFIDENCE_THRESHOLD, feature_contributions
from football_predictor.sports.data_layer import SPORT_CONFIG, MultiSportDataFetcher, SportmonksError
from football_predictor.bookmakers.sportybet import SportyBetAdapter
from football_predictor.sports.sportscore_client import SportscoreClient
from football_predictor.sports.virtual_engine import VirtualSportsEngine

logger = logging.getLogger(__name__)

router = APIRouter()

# Built once and reused: constructing it sets up the AsyncAnthropic client and
# the five analyst personas, neither of which depend on per-request state.
_committee = PredictionCommittee()

# Maps the committee's home_win/draw/away_win keys to the H/D/A codes used
# elsewhere (Prediction.predicted_outcome, Match.result).
_OUTCOME_CODES = dict(zip(OUTCOME_KEYS, ("H", "D", "A")))

_sportscore_client = SportscoreClient()
_virtual_engine = VirtualSportsEngine()

# Sportscore sport_id values used by the /sportscore/{sport}/upcoming endpoint.
SPORT_IDS: dict[str, int] = {
    "football": 1,
    "basketball": 2,
    "tennis": 3,
    "ice_hockey": 4,
    "volleyball": 5,
    "baseball": 6,
    "cricket": 7,
    "table_tennis": 8,
}


class LlmPredictionRequest(BaseModel):
    match_id: int


class BookingCodeRequest(BaseModel):
    match_id: int | str
    prediction: str
    odds: float
    sport: str = "football"


def _to_prediction_out(
    match: Match, prediction: Prediction | None, teams: dict[int, str], auth: AuthContext
) -> PredictionOut:
    out = PredictionOut(
        match_id=match.id,
        kickoff_utc=match.kickoff_utc,
        home_team=teams[match.home_team_id],
        away_team=teams[match.away_team_id],
    )
    if prediction is not None:
        out.predicted_outcome = prediction.predicted_outcome
        out.confidence = prediction.confidence
        if auth.limits["full_probabilities"]:
            out.p_home = prediction.p_home
            out.p_draw = prediction.p_draw
            out.p_away = prediction.p_away
    return out


def _effective_threshold(auth: AuthContext, min_confidence: float | None) -> float | None:
    """The confidence floor a prediction must clear to be surfaced to this caller.

    `None` means "no filtering" (free tier: raw predictions). Elite callers may
    override the default threshold via `min_confidence`; other tiers cannot.
    """
    if auth.limits["custom_threshold"] and min_confidence is not None:
        return min_confidence
    return auth.limits["min_confidence"]


@router.get("/predictions/upcoming", response_model=list[PredictionOut])
def upcoming_predictions(
    days_ahead: int = Query(7, ge=1, le=30, description="How many days ahead to look"),
    min_confidence: float | None = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Elite tier only: override the confidence threshold used to filter predictions",
    ),
    auth: AuthContext = Depends(get_auth_context),
) -> list[PredictionOut]:
    """Predictions for upcoming, unresolved fixtures within the caller's tier horizon.

    Free tier sees every fixture's raw prediction. Pro and elite tiers only see
    predictions whose confidence clears their threshold (see `_effective_threshold`);
    fixtures without a sufficiently confident prediction are omitted entirely.
    """
    horizon = min(days_ahead, auth.limits["fixture_horizon_days"])
    threshold = _effective_threshold(auth, min_confidence)
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    cutoff = now + dt.timedelta(days=horizon)

    with get_session() as session:
        active = session.query(ModelVersion).filter_by(is_active=True).one_or_none()

        matches = (
            session.query(Match)
            .filter(Match.kickoff_utc >= now, Match.kickoff_utc <= cutoff, Match.status == "SCHEDULED")
            .order_by(Match.kickoff_utc)
            .all()
        )
        teams = {t.id: t.canonical_name for t in session.query(Team).all()}

        predictions_by_match: dict[int, Prediction] = {}
        if active and matches:
            match_ids = [m.id for m in matches]
            for p in (
                session.query(Prediction)
                .filter(Prediction.match_id.in_(match_ids), Prediction.model_version_id == active.id)
                .all()
            ):
                predictions_by_match[p.match_id] = p

        results = []
        for m in matches:
            prediction = predictions_by_match.get(m.id)
            if threshold is not None and (prediction is None or prediction.confidence <= threshold):
                continue
            results.append(_to_prediction_out(m, prediction, teams, auth))
        return results


@router.get("/predictions/{match_id}", response_model=PredictionOut)
def prediction_detail(
    match_id: int,
    min_confidence: float | None = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Elite tier only: override the confidence threshold used to filter the prediction",
    ),
    auth: AuthContext = Depends(get_auth_context),
) -> PredictionOut:
    """Single-match prediction detail. Pro/elite tiers also include top feature contributions.

    If the caller's tier filters by confidence and this match's prediction
    doesn't clear the threshold, the prediction fields are withheld (the fixture
    is still returned, as if no prediction had been made yet).
    """
    threshold = _effective_threshold(auth, min_confidence)

    with get_session() as session:
        match = session.get(Match, match_id)
        if match is None:
            raise HTTPException(status_code=404, detail="Match not found")

        active = session.query(ModelVersion).filter_by(is_active=True).one_or_none()
        prediction = None
        if active:
            prediction = (
                session.query(Prediction)
                .filter_by(match_id=match_id, model_version_id=active.id)
                .one_or_none()
            )

        if prediction is not None and threshold is not None and prediction.confidence <= threshold:
            prediction = None

        teams = {t.id: t.canonical_name for t in session.query(Team).all()}
        out = _to_prediction_out(match, prediction, teams, auth)

    if prediction is not None and auth.limits["feature_contributions"]:
        contribs = feature_contributions(match_id)
        if contribs is not None:
            out.top_features = [FeatureContribution(**c) for c in contribs]

    return out


@router.post("/predictions/llm", response_model=LlmPredictionOut)
async def llm_committee_prediction(
    body: LlmPredictionRequest,
    auth: AuthContext = Depends(get_auth_context),
) -> LlmPredictionOut:
    """LLM analyst-committee prediction for a single match (elite tier only).

    Runs the five-agent `PredictionCommittee` (see
    `football_predictor.agents.committee`) for the given match. If the agents
    agree (low variance), their averaged probabilities are returned directly;
    otherwise the response falls back to the active XGBoost model's
    prediction for this match.
    """
    if not auth.limits["llm_committee"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The LLM analyst committee is available on the elite tier only",
        )

    with get_session() as session:
        match = session.get(Match, body.match_id)
        if match is None:
            raise HTTPException(status_code=404, detail="Match not found")

        teams = {t.id: t.canonical_name for t in session.query(Team).all()}
        kickoff_utc = match.kickoff_utc
        home_team = teams[match.home_team_id]
        away_team = teams[match.away_team_id]

    fixture = {
        "match_id": match.id,
        "home_team": home_team,
        "away_team": away_team,
        "datetime": kickoff_utc.isoformat(),
        "season": match.season,
    }

    try:
        result = await _committee.predict(fixture, sport="football")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return LlmPredictionOut(
        match_id=body.match_id,
        kickoff_utc=kickoff_utc,
        home_team=home_team,
        away_team=away_team,
        source=result["source"],
        home_win=result["home_win"],
        draw=result["draw"],
        away_win=result["away_win"],
        variance=result.get("variance"),
        agent_opinions=(
            [AgentOpinionOut(**o) for o in result["agent_opinions"]] if result.get("agent_opinions") else None
        ),
    )


@router.get("/sports/{sport}/upcoming", response_model=list[SportPredictionOut])
async def get_sport_predictions(
    sport: str,
    days_ahead: int = Query(7, ge=1, le=30, description="How many days ahead to look"),
    min_confidence: float | None = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Elite tier only: override the confidence threshold used to filter predictions",
    ),
    auth: AuthContext = Depends(get_auth_context),
) -> list[SportPredictionOut]:
    """Upcoming-fixture predictions for a sport (football, cricket, tennis, f1).

    Football is available on every tier; the other sports require a pro or
    elite API key (`auth.limits["available_sports"]`). Each fixture is run
    through the LLM analyst committee (`football_predictor.agents.committee`),
    which falls back to the active XGBoost model for football fixtures with
    stored features. Results are filtered by the caller's confidence
    threshold (see `_effective_threshold`) and recorded in `SportPrediction`
    so `/sports/leaderboard` can track accuracy once fixtures resolve.
    """
    if sport not in SPORT_CONFIG:
        raise HTTPException(
            status_code=404, detail=f"Unsupported sport: {sport!r}. Supported: {sorted(SPORT_CONFIG)}"
        )
    if sport not in auth.limits["available_sports"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"The {sport} market requires a pro or elite tier API key",
        )

    horizon = min(days_ahead, auth.limits["fixture_horizon_days"])
    threshold = _effective_threshold(auth, min_confidence)
    today = dt.date.today()

    fetcher = MultiSportDataFetcher()
    try:
        fixtures = fetcher.fetch_fixtures(sport, today, today + dt.timedelta(days=horizon))
    except SportmonksError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        fetcher.close()

    p1_key, p2_key = SPORT_CONFIG[sport]["participant_keys"]
    results: list[SportPredictionOut] = []
    to_record: list[dict] = []

    for fixture in fixtures:
        try:
            kickoff_utc = dt.datetime.fromisoformat(fixture["datetime"])
        except (KeyError, ValueError):
            logger.warning("Skipping %s fixture with unparseable datetime: %r", sport, fixture.get("datetime"))
            continue

        try:
            estimate = await _committee.predict(fixture, sport=sport)
        except RuntimeError as exc:
            logger.warning("No prediction available for %s fixture %s: %s", sport, fixture.get("external_id"), exc)
            continue

        probs = {key: estimate[key] for key in OUTCOME_KEYS}
        top_key = max(probs, key=probs.get)
        predicted_outcome = _OUTCOME_CODES[top_key]
        confidence = probs[top_key]

        if threshold is not None and confidence <= threshold:
            continue

        out = SportPredictionOut(
            sport=sport,
            external_id=fixture["external_id"],
            kickoff_utc=kickoff_utc,
            participants={p1_key: fixture.get(p1_key), p2_key: fixture.get(p2_key)},
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            source=estimate["source"],
        )
        if auth.limits["full_probabilities"]:
            out.home_win = estimate["home_win"]
            out.draw = estimate["draw"]
            out.away_win = estimate["away_win"]

        agent_opinions = estimate.get("agent_opinions")
        if agent_opinions is not None:
            agreeing = sum(
                1 for o in agent_opinions if max(OUTCOME_KEYS, key=lambda k: o[k]) == top_key
            )
            out.consensus = AgentConsensusOut(agreeing=agreeing, total=len(agent_opinions))
            if auth.limits["llm_committee"]:
                out.variance = estimate.get("variance")
                out.agent_opinions = [AgentOpinionOut(**o) for o in agent_opinions]

        results.append(out)
        to_record.append(
            {
                "external_id": fixture["external_id"],
                "kickoff_utc": kickoff_utc,
                "p_home": estimate["home_win"],
                "p_draw": estimate["draw"],
                "p_away": estimate["away_win"],
                "predicted_outcome": predicted_outcome,
                "confidence": confidence,
                "source": estimate["source"],
            }
        )

    if to_record:
        with get_session() as session:
            existing = {
                row.external_id: row
                for row in session.query(SportPrediction)
                .filter(
                    SportPrediction.sport == sport,
                    SportPrediction.external_id.in_([r["external_id"] for r in to_record]),
                )
                .all()
            }
            for rec in to_record:
                row = existing.get(rec["external_id"])
                if row is None:
                    session.add(SportPrediction(sport=sport, **rec))
                else:
                    for field, value in rec.items():
                        setattr(row, field, value)

    return results


def _sport_tier_bucket(tier: str, threshold: float, rows: list[SportPrediction]) -> TierAccuracyOut:
    subset = [r for r in rows if r.confidence > threshold]
    n = len(subset)
    accuracy = (sum(1 for r in subset if r.correct) / n) if n else 0.0
    coverage_pct = (n / len(rows) * 100) if rows else 0.0
    return TierAccuracyOut(tier=tier, min_confidence=threshold, n_samples=n, accuracy=accuracy, coverage_pct=coverage_pct)


@router.get("/sports/leaderboard", response_model=SportLeaderboardOut)
def sport_accuracy_leaderboard(
    min_confidence: float | None = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Elite tier only: override the confidence threshold used for the 'elite' bucket",
    ),
    auth: AuthContext = Depends(get_auth_context),
) -> SportLeaderboardOut:
    """Resolved-prediction accuracy per sport, bucketed by tier confidence threshold.

    Sports are ordered by overall accuracy (highest first), so callers can see
    which markets the system handles best. Rows come from `SportPrediction`,
    written by `/sports/{sport}/upcoming` and resolved (actual_outcome/correct
    filled in) once a fixture is played; sports with no resolved predictions
    yet show zero samples in every tier.
    """
    elite_threshold = DEFAULT_CONFIDENCE_THRESHOLD
    if auth.limits["custom_threshold"] and min_confidence is not None:
        elite_threshold = min_confidence

    with get_session() as session:
        rows = session.query(SportPrediction).filter(SportPrediction.correct.isnot(None)).all()

    by_sport: dict[str, list[SportPrediction]] = defaultdict(list)
    for row in rows:
        by_sport[row.sport].append(row)

    entries = []
    for sport in SPORT_CONFIG:
        sport_rows = by_sport.get(sport, [])
        n = len(sport_rows)
        overall_accuracy = (sum(1 for r in sport_rows if r.correct) / n) if n else 0.0
        tiers = {
            "free": _sport_tier_bucket("free", 0.0, sport_rows),
            "pro": _sport_tier_bucket("pro", DEFAULT_CONFIDENCE_THRESHOLD, sport_rows),
            "elite": _sport_tier_bucket("elite", elite_threshold, sport_rows),
        }
        entries.append(
            SportLeaderboardEntryOut(sport=sport, n_resolved=n, overall_accuracy=overall_accuracy, tiers=tiers)
        )

    entries.sort(key=lambda e: e.overall_accuracy, reverse=True)
    return SportLeaderboardOut(sports=entries)


@router.get("/sports/{sport}/calibration", response_model=SportCalibrationOut)
def sport_calibration_curve(sport: str, auth: AuthContext = Depends(get_auth_context)) -> SportCalibrationOut:
    """Reliability curve for a sport's resolved LLM-committee predictions (elite tier only).

    Built from `SportPrediction` rows recorded by `/sports/{sport}/upcoming` and
    later resolved with `actual_outcome`/`correct`. Returns an empty `curves`
    dict and `plot=None` if no resolved predictions exist yet for this sport.
    """
    if sport not in SPORT_CONFIG:
        raise HTTPException(
            status_code=404, detail=f"Unsupported sport: {sport!r}. Supported: {sorted(SPORT_CONFIG)}"
        )
    if not auth.limits["calibration_access"]:
        raise HTTPException(status_code=403, detail="Calibration data requires an elite tier API key")

    with get_session() as session:
        rows = (
            session.query(SportPrediction)
            .filter(SportPrediction.sport == sport, SportPrediction.correct.isnot(None))
            .all()
        )

    if not rows:
        return SportCalibrationOut(sport=sport, n_resolved=0, curves={}, plot=None)

    outcome_idx = {"H": 0, "D": 1, "A": 2}
    y_true_idx = np.array([outcome_idx[r.actual_outcome] for r in rows])
    proba = np.array([[r.p_home, r.p_draw, r.p_away] for r in rows])

    curves = calibration_curve_data(y_true_idx, proba)
    return SportCalibrationOut(sport=sport, n_resolved=len(rows), curves=curves, plot=calibration_curve_figure(curves))


# ---------------------------------------------------------------------------
# Sportscore helpers
# ---------------------------------------------------------------------------

def _normalize_sportscore_fixture(raw: dict, sport: str) -> dict:
    """Convert a Sportscore API event dict to the fixture format expected by the committee."""
    home = raw.get("home_team") or {}
    away = raw.get("away_team") or {}
    home_name = home.get("name", "") if isinstance(home, dict) else str(home)
    away_name = away.get("name", "") if isinstance(away, dict) else str(away)
    return {
        "external_id": f"sportscore:{raw.get('id', '')}",
        "home_team": home_name,
        "away_team": away_name,
        "datetime": raw.get("start_at") or raw.get("start_date") or "",
        "sport": sport,
        "league": (raw.get("league") or {}).get("name", ""),
    }


async def _run_ensemble(fixture: dict, sport: str) -> dict:
    """Run the LLM committee on a fixture and return its raw result dict."""
    return await _committee.predict(fixture, sport=sport)


# ---------------------------------------------------------------------------
# Sportscore-backed upcoming predictions
# ---------------------------------------------------------------------------

@router.get("/sportscore/{sport_name}/upcoming", response_model=list[SportPredictionOut])
async def sportscore_upcoming(
    sport_name: str,
    days_ahead: int = Query(7, ge=1, le=30),
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
    auth: AuthContext = Depends(get_auth_context),
) -> list[SportPredictionOut]:
    """Upcoming predictions sourced from Sportscore (RapidAPI) – mock version (no LLM)."""
    sport_id = SPORT_IDS.get(sport_name)
    if sport_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown sport {sport_name!r}. Supported: {sorted(SPORT_IDS)}",
        )

    if sport_name != "football" and sport_name not in auth.limits.get("available_sports", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"The {sport_name} market requires a pro or elite tier API key",
        )

    horizon = min(days_ahead, auth.limits["fixture_horizon_days"])
    date_from = dt.date.today().isoformat()
    date_to = (dt.date.today() + dt.timedelta(days=horizon)).isoformat()

    try:
        raw_response = _sportscore_client.get_fixtures(sport_id, date_from=date_from, date_to=date_to)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Sportscore API error: {exc}") from exc

    raw_fixtures = raw_response.get("data") or []
    results = []

    for raw in raw_fixtures:
        fixture = _normalize_sportscore_fixture(raw, sport_name)

        try:
            kickoff_utc = dt.datetime.fromisoformat(fixture["datetime"])
        except (ValueError, TypeError):
            logger.warning("Skipping sportscore fixture with unparseable datetime: %r", fixture.get("datetime"))
            continue

        # --- MOCK PREDICTION (no LLM call) ---
        import random
        estimate = {
            "home_win": random.uniform(0.4, 0.7),
            "draw": random.uniform(0.1, 0.3),
            "away_win": random.uniform(0.1, 0.3),
            "source": "mock",
            "agent_opinions": None,
        }
        # ----------------------------------------

        probs = {key: estimate[key] for key in OUTCOME_KEYS}
        top_key = max(probs, key=probs.get)
        predicted_outcome = _OUTCOME_CODES[top_key]
        confidence = probs[top_key]

        out = SportPredictionOut(
            sport=sport_name,
            external_id=fixture["external_id"],
            kickoff_utc=kickoff_utc,
            participants={"home_team": fixture["home_team"], "away_team": fixture["away_team"]},
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            source=estimate["source"],
        )
        if auth.limits["full_probabilities"]:
            out.home_win = estimate["home_win"]
            out.draw = estimate["draw"]
            out.away_win = estimate["away_win"]

        results.append(out)

    return results

# ---------------------------------------------------------------------------
# Virtual sports predictions
# ---------------------------------------------------------------------------

@router.get("/virtual/{sport_name}/upcoming", response_model=list[SportPredictionOut])
def virtual_upcoming(
    sport_name: str,
    num: int = Query(10, ge=1, le=50, description="Number of virtual fixtures to generate"),
    auth: AuthContext = Depends(get_auth_context),
) -> list[SportPredictionOut]:
    """Instantly-generated virtual-sport predictions (no external API call needed).

    Generates `num` synthetic fixtures for `sport_name` using `VirtualSportsEngine`,
    which applies league base rates and streak-adjusted momentum. Results are
    returned in `SportPredictionOut` format but are NOT persisted to the database
    (virtual fixtures have no real external_id to resolve against).
    """
    league_key = f"virtual_{sport_name}"
    if league_key not in _virtual_engine.leagues:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown virtual sport {sport_name!r}. Supported: {[k.removeprefix('virtual_') for k in _virtual_engine.leagues]}",
        )

    simulated = _virtual_engine.simulate_round(sport=league_key, num=num)
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)

    results: list[SportPredictionOut] = []
    for i, item in enumerate(simulated):
        pred = item["prediction"]
        probs = pred["probabilities"]
        top_key = max(probs, key=probs.get)
        outcome_map = {"home_win": "H", "draw": "D", "away_win": "A"}
        predicted_outcome = outcome_map[top_key]
        confidence = pred["confidence"]
        kickoff_utc = now + dt.timedelta(minutes=i * 5)

        out = SportPredictionOut(
            sport=league_key,
            external_id=item["id"],
            kickoff_utc=kickoff_utc,
            participants={"home_team": item["home_team"], "away_team": item["away_team"]},
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            source="virtual_engine",
        )
        if auth.limits["full_probabilities"]:
            out.home_win = probs["home_win"]
            out.draw = probs["draw"]
            out.away_win = probs["away_win"]

        results.append(out)

    return results


# ---------------------------------------------------------------------------
# Bookmaker integration — SportyBet booking codes
# ---------------------------------------------------------------------------

@router.post("/bookmaker/sportybet/convert")
def convert_to_sportybet_code(
    body: BookingCodeRequest,
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Convert a prediction into a SportyBet booking code (pro/elite only).

    Generates a deterministic booking code from `match_id`, `prediction`,
    `odds`, and `sport`. In production this would call the SportyBet bet-slip
    API; the current implementation produces a stable mock code suitable for
    demos and integration testing.
    """
    if auth.tier not in ("pro", "elite"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Booking code generation requires a pro or elite tier API key",
        )
    code = SportyBetAdapter.generate_booking_code(
        body.match_id, body.prediction, body.odds, body.sport
    )
    return {"booking_code": code}

# --- Bet9ja Booking Code ---
@router.post("/bookmaker/bet9ja/convert")
def convert_to_bet9ja_code(
    body: BookingCodeRequest,
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    if auth.tier not in ("pro", "elite"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Booking code generation requires a pro or elite tier API key",
        )
    code = Bet9jaAdapter.generate_booking_code(
        body.match_id, body.prediction, body.odds, body.sport
    )
    return {"booking_code": code}

# --- 1xBet Booking Code ---
@router.post("/bookmaker/onexbet/convert")
def convert_to_onexbet_code(
    body: BookingCodeRequest,
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    if auth.tier not in ("pro", "elite"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Booking code generation requires a pro or elite tier API key",
        )
    code = OnexBetAdapter.generate_booking_code(
        body.match_id, body.prediction, body.odds, body.sport
    )
    return {"booking_code": code}
