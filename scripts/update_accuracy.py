#!/usr/bin/env python
"""Score predictions for matches that have finished since the last run."""
from __future__ import annotations

import logging

from football_predictor.models.accuracy import update_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    n = update_accuracy()
    print(f"Scored {n} newly-finished predictions")
