#!/usr/bin/env python
"""Build the feature matrix for all matches and persist it to MatchFeatures."""
from __future__ import annotations

import logging

from football_predictor.features.pipeline import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    n = run()
    print(f"Built/updated features for {n} matches")
