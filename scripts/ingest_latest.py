#!/usr/bin/env python
"""Daily refresh: current-season results/xG, upcoming fixtures, weather.

If the primary competition (PL) has no upcoming fixtures in the requested
window (e.g. its summer off-season), also checks fallback leagues
(see football_data_org.FALLBACK_LEAGUE_NAMES) for upcoming fixtures.

Usage:
    python scripts/ingest_latest.py [--days-ahead N] [--days-back N] [--force-recent]
"""
from __future__ import annotations

import argparse
import logging

from football_predictor.ingestion.pipeline import ingest_latest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-ahead", type=int, default=14)
    parser.add_argument("--days-back", type=int, default=7)
    parser.add_argument(
        "--force-recent",
        action="store_true",
        help="Skip the historical-style current-season CSV/xG backfill; only refresh "
        "fixtures, results, fallback leagues, and weather.",
    )
    args = parser.parse_args()

    summary = ingest_latest(days_ahead=args.days_ahead, days_back=args.days_back, force_recent=args.force_recent)
    print(summary)
