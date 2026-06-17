#!/usr/bin/env python
"""Generate predictions for all unresolved matches using the active model version."""
from __future__ import annotations

import logging

from football_predictor.models.predict import predict_matches

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    n = predict_matches()
    print(f"Wrote predictions for {n} matches")
