#!/usr/bin/env python
"""Batched, resumable weather backfill (handles Open-Meteo rate limits).

Splits all matches into batches of 50, sleeps 10s between batches, retries
HTTP 429s with exponential backoff, and records per-match attempts so a run
can be killed and re-run later, picking up where it left off.

Usage:
    python scripts/backfill_weather_batched.py
"""
from __future__ import annotations

import logging

from football_predictor.ingestion.weather_backfill import run_batched_backfill

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    summary = run_batched_backfill()
    print(summary)
