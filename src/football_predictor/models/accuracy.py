"""Score predictions for matches that have since finished.

For every `Prediction` whose match is now `FINISHED` and that doesn't yet have
a `PredictionResult`, records the actual outcome plus per-prediction Brier
score and log loss. This is what the accuracy dashboard/API endpoints read.
"""
from __future__ import annotations

import logging
import math

from football_predictor.db.models import Match, Prediction, PredictionResult
from football_predictor.db.session import get_session

logger = logging.getLogger(__name__)

_LOG_LOSS_EPS = 1e-15


def brier_score(p_home: float, p_draw: float, p_away: float, actual: str) -> float:
    onehot = {"H": (1.0, 0.0, 0.0), "D": (0.0, 1.0, 0.0), "A": (0.0, 0.0, 1.0)}[actual]
    return (p_home - onehot[0]) ** 2 + (p_draw - onehot[1]) ** 2 + (p_away - onehot[2]) ** 2


def prediction_log_loss(p_home: float, p_draw: float, p_away: float, actual: str) -> float:
    p_actual = {"H": p_home, "D": p_draw, "A": p_away}[actual]
    p_actual = min(max(p_actual, _LOG_LOSS_EPS), 1 - _LOG_LOSS_EPS)
    return -math.log(p_actual)


def update_accuracy() -> int:
    """Score newly-finished matches' predictions. Returns the number scored."""
    count = 0
    with get_session() as session:
        already_scored = {pid for (pid,) in session.query(PredictionResult.prediction_id).all()}

        rows = (
            session.query(Prediction, Match)
            .join(Match, Match.id == Prediction.match_id)
            .filter(Match.status == "FINISHED")
            .all()
        )

        for prediction, match in rows:
            if prediction.id in already_scored:
                continue
            actual = match.result
            if actual is None:
                continue

            session.add(
                PredictionResult(
                    prediction_id=prediction.id,
                    actual_outcome=actual,
                    correct=prediction.predicted_outcome == actual,
                    brier_score=brier_score(prediction.p_home, prediction.p_draw, prediction.p_away, actual),
                    log_loss=prediction_log_loss(prediction.p_home, prediction.p_draw, prediction.p_away, actual),
                )
            )
            count += 1

    logger.info("Scored %d newly-finished predictions", count)
    return count
