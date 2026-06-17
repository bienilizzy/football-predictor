"""Tests for the confidence-threshold gating logic (no DB/model required)."""
from __future__ import annotations

import numpy as np

from football_predictor.models.predict import DEFAULT_CONFIDENCE_THRESHOLD, _classify_with_threshold

LABELS = ["H", "D", "A"]


def test_classify_above_threshold_is_accepted():
    p = np.array([0.92, 0.05, 0.03])
    outcome, confidence, accepted = _classify_with_threshold(p, LABELS, DEFAULT_CONFIDENCE_THRESHOLD)

    assert outcome == "H"
    assert confidence == 0.92
    assert accepted is True


def test_classify_below_threshold_is_rejected():
    p = np.array([0.5, 0.3, 0.2])
    outcome, confidence, accepted = _classify_with_threshold(p, LABELS, DEFAULT_CONFIDENCE_THRESHOLD)

    assert outcome == "H"
    assert confidence == 0.5
    assert accepted is False


def test_classify_at_threshold_is_rejected():
    """Threshold is a strict `>` so a probability exactly at the threshold is withheld."""
    p = np.array([0.88, 0.07, 0.05])
    _, _, accepted = _classify_with_threshold(p, LABELS, 0.88)

    assert accepted is False


def test_coverage_pct_over_a_batch():
    proba = np.array(
        [
            [0.92, 0.05, 0.03],  # accepted
            [0.50, 0.30, 0.20],  # rejected
            [0.05, 0.05, 0.90],  # accepted
            [0.40, 0.35, 0.25],  # rejected
        ]
    )

    results = [_classify_with_threshold(p, LABELS, DEFAULT_CONFIDENCE_THRESHOLD) for p in proba]
    n_accepted = sum(1 for _, _, accepted in results if accepted)
    coverage_pct = n_accepted / len(proba) * 100

    assert n_accepted == 2
    assert coverage_pct == 50.0
