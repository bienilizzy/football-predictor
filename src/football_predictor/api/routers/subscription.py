"""Subscription tier catalog and the caller's current plan/usage."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Security

from football_predictor.api.auth import TIER_LIMITS, AuthContext, api_key_header, get_auth_context, hash_key
from football_predictor.api.schemas import SubscriptionStatusOut, SubscriptionTierOut
from football_predictor.db.models import ApiKey
from football_predictor.db.session import get_session

router = APIRouter()

# Marketing/pricing copy per tier. Headline accuracy figures describe what each
# tier's prediction feed achieves on the active model's held-out test set -- see
# /api/v1/accuracy/by_tier for the live numbers behind these claims.
TIER_CATALOG: dict[str, dict] = {
    "free": {
        "display_name": "Free",
        "monthly_price_usd": 0,
        "description": "Raw, unfiltered predictions for every upcoming fixture.",
        "headline_accuracy": "~55% (all predictions)",
    },
    "pro": {
        "display_name": "Pro",
        "monthly_price_usd": 30,
        "description": "Predictions filtered to >88% model confidence, plus full "
        "probabilities, feature contributions, and accuracy history.",
        "headline_accuracy": "~85% (confidence > 88%)",
    },
    "elite": {
        "display_name": "Elite",
        "monthly_price_usd": 100,
        "description": "Everything in Pro, plus live calibration data, a "
        "caller-defined confidence threshold, and the highest fixture horizon "
        "and API quota.",
        "headline_accuracy": "Custom threshold (you choose the trade-off)",
    },
}


@router.get("/subscription/tiers", response_model=list[SubscriptionTierOut])
def list_tiers() -> list[SubscriptionTierOut]:
    """Public catalog of subscription tiers, pricing, and capabilities."""
    return [
        SubscriptionTierOut(tier=tier, limits=TIER_LIMITS[tier], **catalog_entry)
        for tier, catalog_entry in TIER_CATALOG.items()
    ]


@router.get("/subscription/me", response_model=SubscriptionStatusOut)
def my_subscription(
    auth: AuthContext = Depends(get_auth_context),
    api_key: str = Security(api_key_header),
) -> SubscriptionStatusOut:
    """The caller's current tier, capabilities, and quota usage."""
    with get_session() as session:
        record = session.query(ApiKey).filter_by(key_hash=hash_key(api_key)).one()
        daily_quota = record.daily_quota
        requests_today = record.requests_today

    return SubscriptionStatusOut(
        tier=auth.tier,
        owner_label=auth.owner_label,
        limits=auth.limits,
        daily_quota=daily_quota,
        requests_today=requests_today,
        **TIER_CATALOG[auth.tier],
    )
