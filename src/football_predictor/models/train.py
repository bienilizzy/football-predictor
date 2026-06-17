"""Train an XGBoost multiclass (H/D/A) classifier with probability calibration.

Uses a chronological 3-way split (train / calibration / test) so that:
  - the booster never sees calibration or test matches during training,
  - the calibrator (CalibratedClassifierCV over a FrozenEstimator) never sees test matches,
  - reported metrics come from a genuinely held-out, most-recent slice of matches.
"""
from __future__ import annotations

import datetime as dt
import logging

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from config.settings import settings
from football_predictor.db.models import Match, MatchFeatures, ModelVersion
from football_predictor.db.session import get_session
from football_predictor.models.evaluation import evaluate

logger = logging.getLogger(__name__)

OUTCOME_LABELS = ["H", "D", "A"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(OUTCOME_LABELS)}

NON_FEATURE_COLUMNS = {"match_id", "kickoff_utc", "target"}

MIN_TRAINING_MATCHES = 50


def load_training_data() -> pd.DataFrame:
    """All matches with a known result, as a wide feature DataFrame ordered by kickoff."""
    with get_session() as session:
        rows = (
            session.query(MatchFeatures, Match.kickoff_utc)
            .join(Match, Match.id == MatchFeatures.match_id)
            .filter(MatchFeatures.target.isnot(None))
            .order_by(Match.kickoff_utc)
            .all()
        )

    records = []
    for mf, kickoff in rows:
        record = dict(mf.features)
        record["match_id"] = mf.match_id
        record["kickoff_utc"] = kickoff
        record["target"] = mf.target
        records.append(record)
    return pd.DataFrame(records)


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.7, calib_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a kickoff-ordered DataFrame into (train, calibration, test)."""
    n = len(df)
    train_end = int(n * train_frac)
    calib_end = int(n * (train_frac + calib_frac))
    return df.iloc[:train_end], df.iloc[train_end:calib_end], df.iloc[calib_end:]


def train_model(
    train_frac: float = 0.7,
    calib_frac: float = 0.15,
    calibration_method: str = "sigmoid",
    calibration_cv: int = 5,
    model_name: str | None = None,
) -> dict:
    """Train, calibrate, evaluate, persist the model artifact, and record a ModelVersion.

    Returns a dict with the new model's id, name, artifact path, and test-set metrics.
    """
    df = load_training_data()
    if len(df) < MIN_TRAINING_MATCHES:
        raise RuntimeError(
            f"Only {len(df)} labeled matches available (need at least "
            f"{MIN_TRAINING_MATCHES}). Run scripts/ingest_historical.py and "
            "scripts/build_features.py first."
        )

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    train_df, calib_df, test_df = chronological_split(df, train_frac, calib_frac)
    logger.info(
        "Split %d matches into train=%d calib=%d test=%d",
        len(df), len(train_df), len(calib_df), len(test_df),
    )

    X_train, y_train = train_df[feature_cols], train_df["target"].map(LABEL_TO_IDX)
    X_calib, y_calib = calib_df[feature_cols], calib_df["target"].map(LABEL_TO_IDX)
    X_test, y_test = test_df[feature_cols], test_df["target"].map(LABEL_TO_IDX)

    base_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
    )
    base_model.fit(X_train, y_train)

    # FrozenEstimator prevents CalibratedClassifierCV from refitting the booster;
    # this is the modern replacement for the removed `cv="prefit"` option. With
    # `ensemble=True`, `cv` splits X_calib into folds and fits one calibrator per
    # fold (all wrapping the same frozen booster), averaging their outputs at
    # predict time -- this reduces variance from any single calibration split.
    # `method="sigmoid"` (Platt scaling, 2 params/class) is far less prone to
    # overfitting the ~170-row calibration set than "isotonic", which previously
    # produced overconfident, poorly-calibrated probabilities (log_loss > ln(3)).
    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(base_model), method=calibration_method, cv=calibration_cv, ensemble=True
    )
    calibrated_model.fit(X_calib, y_calib)

    test_proba = calibrated_model.predict_proba(X_test)
    metrics = evaluate(y_test.to_numpy(), test_proba, labels=tuple(OUTCOME_LABELS))
    metrics.update(
        {
            "train_size": len(train_df),
            "calib_size": len(calib_df),
            "test_size": len(test_df),
            "calibration_method": calibration_method,
            "calibration_cv": calibration_cv,
        }
    )

    name = model_name or f"xgb_{dt.datetime.now(dt.UTC).replace(tzinfo=None):%Y%m%d_%H%M%S}"
    artifact_path = settings.model_dir / f"{name}.joblib"
    joblib.dump(
        {
            "model": calibrated_model,
            "feature_columns": feature_cols,
            "outcome_labels": OUTCOME_LABELS,
        },
        artifact_path,
    )

    with get_session() as session:
        session.query(ModelVersion).filter(ModelVersion.is_active.is_(True)).update(
            {ModelVersion.is_active: False}
        )
        version = ModelVersion(
            name=name,
            feature_names=feature_cols,
            metrics=metrics,
            artifact_path=str(artifact_path),
            is_active=True,
        )
        session.add(version)
        session.flush()
        version_id = version.id

    logger.info(
        "Trained model '%s': accuracy=%.3f log_loss=%.3f brier=%.3f (test_size=%d)",
        name, metrics["accuracy"], metrics["log_loss"], metrics["brier_score"], metrics["test_size"],
    )

    return {"id": version_id, "name": name, "artifact_path": str(artifact_path), "metrics": metrics}
