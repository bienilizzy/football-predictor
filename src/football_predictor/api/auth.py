"""API key authentication, tier resolution, and daily quota enforcement.

Tiers are looked up from the `ApiKey` table by SHA-256 hash of the raw key
(see db/init_db.py for the seeded demo keys). Quotas reset once per UTC day.
"""
from __future__ import annotations

import datetime as dt
import hashlib

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from football_predictor.db.models import ApiKey
from football_predictor.db.session import get_session
from football_predictor.models.predict import DEFAULT_CONFIDENCE_THRESHOLD

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Capabilities and limits per subscription tier.
#   free:  raw, unfiltered predictions (min_confidence=None -> nothing withheld).
#       Only the football market is available.
#   pro ($30/mo):  predictions filtered to confidence > DEFAULT_CONFIDENCE_THRESHOLD,
#       plus access to the cricket/tennis/f1 markets.
#   elite ($100/mo): same filtering as pro but the threshold can be overridden per
#       request (custom_threshold), plus access to live calibration data, the
#       LLM analyst committee, and the broadest fixture horizon/quota.
TIER_LIMITS: dict[str, dict] = {
    "free": {
        "fixture_horizon_days": 3,
        "full_probabilities": False,
        "feature_contributions": False,
        "accuracy_history": False,
        "min_confidence": None,
        "custom_threshold": False,
        "calibration_access": False,
        "llm_committee": False,
        "available_sports": ["football", "cricket", "basketball", "tennis", "icehockey", "volleyball"],
    },
    "pro": {
        "fixture_horizon_days": 14,
        "full_probabilities": True,
        "feature_contributions": True,
        "accuracy_history": True,
        "min_confidence": DEFAULT_CONFIDENCE_THRESHOLD,
        "custom_threshold": False,
        "calibration_access": False,
        "llm_committee": False,
        "available_sports": ["football", "cricket", "tennis", "f1", "basketball","icehockey", "volleyball"],
    },
    "elite": {
        "fixture_horizon_days": 30,
        "full_probabilities": True,
        "feature_contributions": True,
        "accuracy_history": True,
        "min_confidence": DEFAULT_CONFIDENCE_THRESHOLD,
        "custom_threshold": True,
        "calibration_access": True,
        "llm_committee": True,
        "available_sports": ["football", "cricket", "tennis", "f1", "basketball", "icehockey", "volleyball"],
    },
}


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class AuthContext:
    """Resolved identity for the current request."""

    def __init__(self, tier: str, owner_label: str) -> None:
        self.tier = tier
        self.owner_label = owner_label
        self.limits = TIER_LIMITS[tier]


def get_auth_context(api_key: str | None = Security(api_key_header)) -> AuthContext:
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-API-Key header")

    key_hash = hash_key(api_key)

    with get_session() as session:
        record = session.query(ApiKey).filter_by(key_hash=key_hash).one_or_none()
        if record is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

        now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
        if now >= record.quota_reset_at:
            record.requests_today = 0
            record.quota_reset_at = now.replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)

        if record.requests_today >= record.daily_quota:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Daily quota exceeded")

        record.requests_today += 1
        tier, owner_label = record.tier, record.owner_label

    return AuthContext(tier=tier, owner_label=owner_label)
