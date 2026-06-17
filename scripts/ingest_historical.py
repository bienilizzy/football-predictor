#!/usr/bin/env python
"""Backfill historical seasons: results, referees, cards, shots, xG, weather.

Usage:
    python scripts/ingest_historical.py [season_code ...]

If no season codes are given, uses HISTORICAL_SEASONS from .env (e.g. 2223 2324 2425).
"""
from __future__ import annotations

import logging
import sys

from football_predictor.ingestion.pipeline import ingest_historical

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    seasons = sys.argv[1:] or None
    summary = ingest_historical(seasons=seasons)
    print(summary)
