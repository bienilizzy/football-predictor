"""Tests for the calibration wrapper and evaluation metrics (no DB required)."""
from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from football_predictor.models.evaluation import evaluate, multiclass_brier_score


def _synthetic_dataset(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    logits = X[:, 0] * 2 + rng.normal(scale=0.5, size=n)
    y = np.digitize(logits, bins=[-0.5, 0.5])  # classes 0, 1, 2
    return X, y


def test_calibrated_probabilities_sum_to_one():
    X, y = _synthetic_dataset()
    X_train, y_train = X[:200], y[:200]
    X_calib, y_calib = X[200:], y[200:]

    base = xgb.XGBClassifier(objective="multi:softprob", n_estimators=20, max_depth=2)
    base.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(X_calib, y_calib)

    proba = calibrated.predict_proba(X_calib)
    assert proba.shape == (len(X_calib), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_multiclass_brier_score_perfect_predictions():
    y_true = np.array([0, 1, 2])
    perfect = np.eye(3)
    assert multiclass_brier_score(y_true, perfect) == pytest.approx(0.0)


def test_multiclass_brier_score_uniform_predictions():
    y_true = np.array([0, 1, 2])
    uniform = np.full((3, 3), 1 / 3)
    # Per row: two classes off by (1/3)^2 and the true class off by (1 - 1/3)^2.
    expected = 2 * (1 / 3) ** 2 + (2 / 3) ** 2
    assert multiclass_brier_score(y_true, uniform) == pytest.approx(expected)


def test_evaluate_returns_expected_structure():
    y_true = np.tile([0, 1, 2], 5)
    rng = np.random.default_rng(1)
    proba = rng.dirichlet(alpha=[2, 2, 2], size=len(y_true))

    result = evaluate(y_true, proba, n_bins=5)

    assert result.keys() >= {"accuracy", "log_loss", "brier_score", "calibration_curve", "n_samples"}
    assert result["n_samples"] == len(y_true)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert set(result["calibration_curve"].keys()) == {"H", "D", "A"}
