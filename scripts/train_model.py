#!/usr/bin/env python
"""Train a new model version on all currently labeled matches and activate it."""
from __future__ import annotations

import logging

from football_predictor.models.train import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    result = train_model()
    metrics = result["metrics"]
    print(f"Trained model '{result['name']}' (id={result['id']})")
    print(f"  artifact: {result['artifact_path']}")
    print(f"  test set: {metrics['test_size']} matches")
    print(f"  accuracy: {metrics['accuracy']:.3f}")
    print(f"  log loss: {metrics['log_loss']:.3f}")
    print(f"  brier:    {metrics['brier_score']:.3f}")
