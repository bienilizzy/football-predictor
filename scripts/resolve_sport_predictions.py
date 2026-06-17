#!/usr/bin/env python
"""Resolve SportPrediction outcomes for past fixtures.

Fetches actual results from football-data.org (for fd_org: fixtures) and
Sportmonks (for sportmonks: fixtures), then writes actual_outcome / correct
so the /sports/leaderboard and /sports/{sport}/calibration endpoints have
data to display.

Run daily after matches have finished:
    python scripts/resolve_sport_predictions.py [--dry-run]
"""
from __future__ import annotations

import argparse
import logging

from football_predictor.models.resolve_sports import resolve_sport_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Fetch results but don't write to DB")
    args = parser.parse_args()

    summary = resolve_sport_predictions(dry_run=args.dry_run)
    print(
        f"Resolved: {summary['resolved']}  "
        f"Skipped: {summary['skipped']}  "
        f"Errors: {summary['errors']}"
    )


if __name__ == "__main__":
    main()
