"""Redis-backed cache for LLM analyst-committee responses.

`PredictionCommittee.committee_estimate` is the expensive part of a
prediction - five parallel Claude calls - and its result for a given fixture
doesn't change within a short window. Repeat requests for the same fixture +
sport (e.g. multiple callers hitting `/sports/{sport}/upcoming` within the
same hour) are served from this cache instead of re-querying the committee.

If Redis is unreachable (e.g. local dev without `docker compose up redis`, or
the test suite), every lookup is treated as a cache miss and every store is a
no-op - the LLM committee still works, just without caching.
"""
from __future__ import annotations

import json
import logging

import redis.asyncio as redis
from redis.exceptions import RedisError

from config.settings import settings

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 60 * 60  # 1 hour
_KEY_PREFIX = "llm_committee"

# Fail fast if Redis is unreachable rather than hanging the request.
_CONNECT_TIMEOUT_SECONDS = 2


class CommitteeResponseCache:
    """Caches `PredictionCommittee.committee_estimate` results in Redis."""

    def __init__(self, redis_url: str | None = None):
        self._client = redis.from_url(
            redis_url or settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=_CONNECT_TIMEOUT_SECONDS,
            socket_timeout=_CONNECT_TIMEOUT_SECONDS,
        )

    @staticmethod
    def _key(sport: str, fixture_id: str) -> str:
        return f"{_KEY_PREFIX}:{sport}:{fixture_id}"

    async def get(self, sport: str, fixture_id: str) -> dict | None:
        """Return the cached committee estimate for this fixture, or `None` on a miss."""
        try:
            payload = await self._client.get(self._key(sport, fixture_id))
        except RedisError as exc:
            logger.debug("Committee cache unavailable (get): %s", exc)
            return None
        return json.loads(payload) if payload is not None else None

    async def set(self, sport: str, fixture_id: str, estimate: dict) -> None:
        """Cache a committee estimate for `CACHE_TTL_SECONDS`."""
        try:
            await self._client.set(self._key(sport, fixture_id), json.dumps(estimate), ex=CACHE_TTL_SECONDS)
        except RedisError as exc:
            logger.debug("Committee cache unavailable (set): %s", exc)
