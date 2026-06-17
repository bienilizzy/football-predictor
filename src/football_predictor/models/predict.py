"""Score matches with the active model and persist results to the Prediction table."""
from __future__ import annotations

import datetime as dt
import logging

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from football_predictor.db.models import MatchFeatures, ModelVersion, Prediction, RejectedPrediction
from football_predictor.db.session import get_session

logger = logging.getLogger(__name__)

# Below this confidence, a prediction is withheld (see `predict_with_rejection`).
DEFAULT_CONFIDENCE_THRESHOLD = 0.88


def load_active_model() -> tuple[int, dict]:
    """Returns (model_version_id, artifact dict with model/feature_columns/outcome_labels)."""
    with get_session() as session:
        version = session.query(ModelVersion).filter_by(is_active=True).one_or_none()
        if version is None:
            raise RuntimeError("No active model found. Run scripts/train_model.py first.")
        artifact_path = version.artifact_path
        version_id = version.id

    artifact = joblib.load(artifact_path)
    return version_id, artifact


def predict_matches(match_ids: list[int] | None = None) -> int:
    """Generate/update predictions for matches that have features but no result yet.

    If `match_ids` is given, restrict to that set (still requires features to exist
    and the match to be unresolved). Returns the number of predictions written.
    """
    version_id, artifact = load_active_model()
    model = artifact["model"]
    feature_cols = artifact["feature_columns"]
    labels: list[str] = artifact["outcome_labels"]
    home_idx, draw_idx, away_idx = labels.index("H"), labels.index("D"), labels.index("A")

    with get_session() as session:
        query = session.query(MatchFeatures).filter(MatchFeatures.target.is_(None))
        if match_ids is not None:
            query = query.filter(MatchFeatures.match_id.in_(match_ids))
        feature_rows = query.all()

        if not feature_rows:
            return 0

        ids = [fr.match_id for fr in feature_rows]
        X = pd.DataFrame([{c: fr.features.get(c, 0.0) for c in feature_cols} for fr in feature_rows])
        proba = model.predict_proba(X)

        existing = {
            p.match_id: p
            for p in session.query(Prediction)
            .filter(Prediction.match_id.in_(ids), Prediction.model_version_id == version_id)
            .all()
        }

        count = 0
        for match_id, p in zip(ids, proba):
            predicted_idx = int(np.argmax(p))
            pred = existing.get(match_id)
            if pred is None:
                pred = Prediction(match_id=match_id, model_version_id=version_id)
                session.add(pred)
            pred.p_home = float(p[home_idx])
            pred.p_draw = float(p[draw_idx])
            pred.p_away = float(p[away_idx])
            pred.predicted_outcome = labels[predicted_idx]
            pred.confidence = float(p[predicted_idx])
            count += 1

    logger.info("Wrote %d predictions using model_version_id=%d", count, version_id)
    return count


def _classify_with_threshold(p: np.ndarray, labels: list[str], threshold: float) -> tuple[str, float, bool]:
    """Returns (predicted_outcome, confidence, accepted) for one probability row."""
    predicted_idx = int(np.argmax(p))
    confidence = float(p[predicted_idx])
    return labels[predicted_idx], confidence, confidence > threshold


def predict_with_rejection(
    match_ids: list[int] | None = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Score matches but only surface predictions the model is confident about.

    For each unresolved match, the predicted probability of every outcome
    (H/D/A) is calculated. A prediction is only included in the result if its
    top probability exceeds `threshold`; matches that don't clear the bar are
    persisted to `RejectedPrediction` instead, for later analysis of where the
    model lacks confidence.

    Returns a summary dict:
        {
            "predictions": [{"match_id", "prediction", "confidence", "predicted_at"}, ...],
            "n_total": int,
            "n_accepted": int,
            "n_rejected": int,
            "coverage_pct": float,  # share of evaluated matches that met the threshold
            "threshold": float,
            "model_version_id": int,
        }
    """
    version_id, artifact = load_active_model()
    model = artifact["model"]
    feature_cols = artifact["feature_columns"]
    labels: list[str] = artifact["outcome_labels"]
    home_idx, draw_idx, away_idx = labels.index("H"), labels.index("D"), labels.index("A")

    with get_session() as session:
        query = session.query(MatchFeatures).filter(MatchFeatures.target.is_(None))
        if match_ids is not None:
            query = query.filter(MatchFeatures.match_id.in_(match_ids))
        feature_rows = query.all()

        if not feature_rows:
            return {
                "predictions": [],
                "n_total": 0,
                "n_accepted": 0,
                "n_rejected": 0,
                "coverage_pct": 0.0,
                "threshold": threshold,
                "model_version_id": version_id,
            }

        ids = [fr.match_id for fr in feature_rows]
        X = pd.DataFrame([{c: fr.features.get(c, 0.0) for c in feature_cols} for fr in feature_rows])
        proba = model.predict_proba(X)

        existing_rejected = {
            r.match_id: r
            for r in session.query(RejectedPrediction)
            .filter(RejectedPrediction.match_id.in_(ids), RejectedPrediction.model_version_id == version_id)
            .all()
        }

        predicted_at = dt.datetime.now(dt.UTC).replace(tzinfo=None)
        accepted: list[dict] = []
        n_rejected = 0

        for match_id, p in zip(ids, proba):
            outcome, confidence, accepted_flag = _classify_with_threshold(p, labels, threshold)
            stale_rejection = existing_rejected.pop(match_id, None)

            if accepted_flag:
                accepted.append(
                    {
                        "match_id": match_id,
                        "prediction": outcome,
                        "confidence": confidence,
                        "predicted_at": predicted_at,
                    }
                )
                if stale_rejection is not None:
                    session.delete(stale_rejection)
            else:
                n_rejected += 1
                rejected = stale_rejection or RejectedPrediction(match_id=match_id, model_version_id=version_id)
                if stale_rejection is None:
                    session.add(rejected)
                rejected.p_home = float(p[home_idx])
                rejected.p_draw = float(p[draw_idx])
                rejected.p_away = float(p[away_idx])
                rejected.predicted_outcome = outcome
                rejected.confidence = confidence
                rejected.threshold = threshold

    n_total = len(ids)
    n_accepted = len(accepted)
    coverage_pct = (n_accepted / n_total * 100) if n_total else 0.0

    logger.info(
        "Confidence-gated predictions: %d/%d matches accepted (coverage %.1f%%) at threshold %.2f",
        n_accepted, n_total, coverage_pct, threshold,
    )

    return {
        "predictions": accepted,
        "n_total": n_total,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "coverage_pct": coverage_pct,
        "threshold": threshold,
        "model_version_id": version_id,
    }


def feature_contributions(match_id: int, top_n: int = 10) -> list[dict] | None:
    """SHAP-style per-feature contributions to the predicted outcome.

    Reads contributions from the underlying XGBoost booster (the single base
    estimator wrapped by CalibratedClassifierCV via FrozenEstimator), for the
    class the calibrated model actually predicts. Returns None if the match
    has no stored feature vector.
    """
    _, artifact = load_active_model()
    model = artifact["model"]
    feature_cols = artifact["feature_columns"]

    with get_session() as session:
        mf = session.get(MatchFeatures, match_id)
        if mf is None:
            return None
        features = mf.features

    X = pd.DataFrame([{c: features.get(c, 0.0) for c in feature_cols}])
    predicted_idx = int(np.argmax(model.predict_proba(X)[0]))

    base_estimator = model.calibrated_classifiers_[0].estimator
    booster = base_estimator.get_booster()
    dmatrix = xgb.DMatrix(X, feature_names=feature_cols)
    contribs = booster.predict(dmatrix, pred_contribs=True)

    class_contribs = contribs[0, predicted_idx, :-1] if contribs.ndim == 3 else contribs[0, :-1]

    rows = list(zip(feature_cols, X.iloc[0].tolist(), class_contribs.tolist()))
    rows.sort(key=lambda r: abs(r[2]), reverse=True)
    return [{"feature": f, "value": v, "contribution": c} for f, v, c in rows[:top_n]]
