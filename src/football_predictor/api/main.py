"""FastAPI application entrypoint.

Run with: uvicorn football_predictor.api.main:app --reload
"""
from __future__ import annotations

import datetime as dt
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from football_predictor.api.routers import accuracy, fixtures, predictions, subscription
from football_predictor.sports.data_layer import SPORT_CONFIG, MultiSportDataFetcher

logger = logging.getLogger(__name__)

CACHE_WARM_DAYS_AHEAD = 7


def _warm_sports_cache() -> None:
    """Pre-populate the SQLite fixture cache for all supported sports."""
    fetcher = MultiSportDataFetcher()
    today = dt.date.today()
    date_to = today + dt.timedelta(days=CACHE_WARM_DAYS_AHEAD)
    try:
        for sport in SPORT_CONFIG:
            try:
                fetcher.fetch_fixtures(sport, today, date_to)
            except Exception:
                logger.exception("Failed to warm fixture cache for sport=%s", sport)
    finally:
        fetcher.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warm_sports_cache()
    yield


app = FastAPI(
    title="Football Predictor API",
    description="Premier League match outcome predictions with tiered access.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(fixtures.router, prefix="/api/v1", tags=["fixtures"])
app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
app.include_router(accuracy.router, prefix="/api/v1", tags=["accuracy"])
app.include_router(subscription.router, prefix="/api/v1", tags=["subscription"])


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}
