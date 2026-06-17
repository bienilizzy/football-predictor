"""Create all tables and seed static reference data (teams, demo API keys)."""
from __future__ import annotations

import datetime as dt
import hashlib

from football_predictor.db.models import ApiKey, Base, Team
from football_predictor.db.session import engine, get_session
from football_predictor.reference_data import PREMIER_LEAGUE_TEAMS

# Demo API keys seeded for local development / dashboard use, one per tier.
DEMO_API_KEYS = {
    "free": ("demo-free-key", 50),
    "pro": ("demo-pro-key", 500),
    "elite": ("demo-elite-key", 5000),
}


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _seed_teams()
    _seed_demo_api_keys()


def _seed_teams() -> None:
    with get_session() as session:
        existing = {t.canonical_name for t in session.query(Team).all()}
        for team_data in PREMIER_LEAGUE_TEAMS:
            if team_data["canonical_name"] not in existing:
                session.add(Team(**team_data))


def _seed_demo_api_keys() -> None:
    tomorrow = dt.datetime.now(dt.UTC).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    ) + dt.timedelta(days=1)

    with get_session() as session:
        existing = {k.key_hash for k in session.query(ApiKey).all()}
        for tier, (raw_key, quota) in DEMO_API_KEYS.items():
            key_hash = _hash_key(raw_key)
            if key_hash not in existing:
                session.add(
                    ApiKey(
                        key_hash=key_hash,
                        owner_label=f"demo-{tier}",
                        tier=tier,
                        daily_quota=quota,
                        requests_today=0,
                        quota_reset_at=tomorrow,
                    )
                )


if __name__ == "__main__":
    init_db()
    print("Database initialized.")
    print("Demo API keys (for local dev / dashboard use):")
    for tier, (raw_key, quota) in DEMO_API_KEYS.items():
        print(f"  {tier:>5}: {raw_key}  (quota={quota}/day)")
