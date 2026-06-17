"""Evaluation metrics for the 3-class (H/D/A) outcome model.

All functions take `y_true_idx` (integer class indices, 0=H, 1=D, 2=A) and
`proba` (an (n_samples, 3) array of predicted probabilities in that same
class order) so they work identically for the raw and calibrated models.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss

OUTCOME_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}


def multiclass_brier_score(y_true_idx: np.ndarray, proba: np.ndarray, n_classes: int = 3) -> float:
    """Mean squared error between one-hot true labels and predicted probabilities,
    summed across classes (the standard multi-class Brier score)."""
    onehot = np.eye(n_classes)[np.asarray(y_true_idx)]
    return float(np.mean(np.sum((np.asarray(proba) - onehot) ** 2, axis=1)))


def calibration_curve_data(
    y_true_idx: np.ndarray,
    proba: np.ndarray,
    labels: tuple[str, ...] = ("H", "D", "A"),
    n_bins: int = 10,
) -> dict[str, dict[str, list[float]]]:
    """One-vs-rest reliability curve per class: mean predicted probability vs
    observed frequency, bucketed into `n_bins` equal-width bins."""
    y_true_idx = np.asarray(y_true_idx)
    result: dict[str, dict[str, list[float]]] = {}
    for i, label in enumerate(labels):
        y_binary = (y_true_idx == i).astype(int)
        prob_true, prob_pred = calibration_curve(
            y_binary, proba[:, i], n_bins=n_bins, strategy="uniform"
        )
        result[label] = {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}
    return result


def calibration_curve_figure(curves: dict[str, dict[str, list[float]]]) -> dict:
    """Reliability-curve plot (perfectly-calibrated diagonal + one trace per
    outcome) as a JSON-serializable Plotly figure spec."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated", line=dict(dash="dash"))
    )
    for outcome, curve in curves.items():
        fig.add_trace(
            go.Scatter(
                x=curve["prob_pred"],
                y=curve["prob_true"],
                mode="lines+markers",
                name=OUTCOME_LABELS.get(outcome, outcome),
            )
        )
    fig.update_layout(
        xaxis=dict(title="Predicted probability", range=[0, 1], tickformat=".0%"),
        yaxis=dict(title="Observed frequency", range=[0, 1], tickformat=".0%"),
        height=450,
    )
    return fig.to_plotly_json()


def per_sample_predictions(y_true_idx: np.ndarray, proba: np.ndarray) -> list[dict]:
    """Per-sample (confidence, correct) pairs for the top predicted class.

    Used to compute confidence-bucketed accuracy (e.g. "accuracy among
    predictions above X% confidence") without re-running inference.
    """
    y_true_idx = np.asarray(y_true_idx)
    y_pred_idx = proba.argmax(axis=1)
    confidences = proba[np.arange(len(proba)), y_pred_idx]
    return [
        {"confidence": float(conf), "correct": bool(pred == true)}
        for conf, pred, true in zip(confidences, y_pred_idx, y_true_idx)
    ]


def evaluate(
    y_true_idx: np.ndarray,
    proba: np.ndarray,
    labels: tuple[str, ...] = ("H", "D", "A"),
    n_bins: int = 10,
) -> dict:
    """Bundle of accuracy, log loss, Brier score, and calibration curve data."""
    y_true_idx = np.asarray(y_true_idx)
    y_pred_idx = proba.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true_idx, y_pred_idx)),
        "log_loss": float(log_loss(y_true_idx, proba, labels=list(range(len(labels))))),
        "brier_score": multiclass_brier_score(y_true_idx, proba, n_classes=len(labels)),
        "calibration_curve": calibration_curve_data(y_true_idx, proba, labels=labels, n_bins=n_bins),
        "n_samples": int(len(y_true_idx)),
        "test_predictions": per_sample_predictions(y_true_idx, proba),
    }
