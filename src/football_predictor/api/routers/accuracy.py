"""Tracked accuracy endpoints for the active model."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from football_predictor.api.auth import AuthContext, get_auth_context
from football_predictor.api.schemas import (
    AccuracyByTierOut,
    AccuracyHistoryEntryOut,
    AccuracySummaryOut,
    CalibrationCurveOut,
    OutcomeAccuracy,
    TierAccuracyOut,
)
from football_predictor.db.models import Match, ModelVersion, Prediction, PredictionResult, Team
from football_predictor.db.session import get_session
from football_predictor.models.evaluation import calibration_curve_figure
from football_predictor.models.predict import DEFAULT_CONFIDENCE_THRESHOLD

router = APIRouter()


@router.get("/accuracy/summary", response_model=AccuracySummaryOut)
def accuracy_summary(auth: AuthContext = Depends(get_auth_context)) -> AccuracySummaryOut:
    """Aggregate accuracy/log-loss/Brier score for the active model's scored predictions."""
    with get_session() as session:
        active = session.query(ModelVersion).filter_by(is_active=True).one_or_none()
        if active is None:
            raise HTTPException(status_code=404, detail="No active model")

        rows = (
            session.query(PredictionResult, Prediction)
            .join(Prediction, Prediction.id == PredictionResult.prediction_id)
            .filter(Prediction.model_version_id == active.id)
            .all()
        )

        if not rows:
            return AccuracySummaryOut(model_name=active.name, n_predictions=0, accuracy=0.0)

        n = len(rows)
        correct = sum(1 for r, _ in rows if r.correct)
        avg_log_loss = sum(r.log_loss for r, _ in rows) / n
        avg_brier = sum(r.brier_score for r, _ in rows) / n

        by_outcome: dict[str, OutcomeAccuracy] | None = None
        if auth.limits["accuracy_history"]:
            by_outcome = {}
            for outcome in ("H", "D", "A"):
                subset = [r for r, p in rows if p.predicted_outcome == outcome]
                if subset:
                    by_outcome[outcome] = OutcomeAccuracy(
                        n_predicted=len(subset),
                        accuracy=sum(1 for r in subset if r.correct) / len(subset),
                    )

        return AccuracySummaryOut(
            model_name=active.name,
            n_predictions=n,
            accuracy=correct / n,
            log_loss=avg_log_loss,
            brier_score=avg_brier,
            by_outcome=by_outcome,
        )


@router.get("/accuracy/calibration", response_model=CalibrationCurveOut)
def calibration_curve(auth: AuthContext = Depends(get_auth_context)) -> CalibrationCurveOut:
    """Reliability-curve data (predicted vs. observed frequency per outcome) for the
    active model's held-out test set, computed during training (elite tier only)."""
    if not auth.limits["calibration_access"]:
        raise HTTPException(status_code=403, detail="Calibration data requires an elite tier API key")

    with get_session() as session:
        active = session.query(ModelVersion).filter_by(is_active=True).one_or_none()
        if active is None:
            raise HTTPException(status_code=404, detail="No active model")

        curves = active.metrics.get("calibration_curve", {})
        return CalibrationCurveOut(
            model_name=active.name,
            test_size=active.metrics.get("test_size", 0),
            curves=curves,
            plot=calibration_curve_figure(curves) if curves else None,
        )


@router.get("/accuracy/history", response_model=list[AccuracyHistoryEntryOut])
def accuracy_history(auth: AuthContext = Depends(get_auth_context)) -> list[AccuracyHistoryEntryOut]:
    """Per-match predicted vs. actual outcomes (pro tier only)."""
    if not auth.limits["accuracy_history"]:
        raise HTTPException(status_code=403, detail="Accuracy history requires a pro tier API key")

    with get_session() as session:
        active = session.query(ModelVersion).filter_by(is_active=True).one_or_none()
        if active is None:
            return []

        rows = (
            session.query(PredictionResult, Prediction, Match)
            .join(Prediction, Prediction.id == PredictionResult.prediction_id)
            .join(Match, Match.id == Prediction.match_id)
            .filter(Prediction.model_version_id == active.id)
            .order_by(Match.kickoff_utc)
            .all()
        )
        teams = {t.id: t.canonical_name for t in session.query(Team).all()}

        return [
            AccuracyHistoryEntryOut(
                match_id=match.id,
                kickoff_utc=match.kickoff_utc,
                home_team=teams[match.home_team_id],
                away_team=teams[match.away_team_id],
                predicted_outcome=pred.predicted_outcome,
                actual_outcome=result.actual_outcome,
                correct=result.correct,
                p_home=pred.p_home,
                p_draw=pred.p_draw,
                p_away=pred.p_away,
            )
            for result, pred, match in rows
        ]


def _tier_bucket(tier: str, threshold: float, test_predictions: list[dict]) -> TierAccuracyOut:
    subset = [tp for tp in test_predictions if tp["confidence"] > threshold]
    n = len(subset)
    accuracy = (sum(1 for tp in subset if tp["correct"]) / n) if n else 0.0
    coverage_pct = (n / len(test_predictions) * 100) if test_predictions else 0.0
    return TierAccuracyOut(tier=tier, min_confidence=threshold, n_samples=n, accuracy=accuracy, coverage_pct=coverage_pct)


@router.get("/accuracy/by_tier", response_model=AccuracyByTierOut)
def accuracy_by_tier(
    min_confidence: float | None = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Elite tier only: recompute the 'elite' bucket at this confidence threshold",
    ),
    auth: AuthContext = Depends(get_auth_context),
) -> AccuracyByTierOut:
    """Confidence-bucketed accuracy on the active model's held-out test set.

    Demonstrates the gap between unfiltered (free), >88%-confidence (pro), and
    elite (caller-chosen threshold) predictions on the same held-out matches.
    """
    with get_session() as session:
        active = session.query(ModelVersion).filter_by(is_active=True).one_or_none()
        if active is None:
            raise HTTPException(status_code=404, detail="No active model")

        model_name = active.name
        test_predictions = active.metrics.get("test_predictions", [])

    elite_threshold = DEFAULT_CONFIDENCE_THRESHOLD
    if auth.limits["custom_threshold"] and min_confidence is not None:
        elite_threshold = min_confidence

    tiers = {
        "free": _tier_bucket("free", 0.0, test_predictions),
        "pro": _tier_bucket("pro", DEFAULT_CONFIDENCE_THRESHOLD, test_predictions),
        "elite": _tier_bucket("elite", elite_threshold, test_predictions),
    }

    return AccuracyByTierOut(model_name=model_name, test_size=len(test_predictions), tiers=tiers)
