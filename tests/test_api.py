"""FastAPI endpoint tests: auth, tier gating, quota enforcement.

Runs against the isolated, file-based SQLite test DB set up in conftest.py
(no model has been trained in this DB, so accuracy endpoints exercise their
"no active model" branches).
"""
from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from football_predictor.api.main import app
from football_predictor.db.models import ApiKey, Match, ModelVersion, Team
from football_predictor.db.session import get_session

client = TestClient(app)

FREE_KEY = "demo-free-key"
PRO_KEY = "demo-pro-key"
ELITE_KEY = "demo-elite-key"


def _auth(key: str) -> dict[str, str]:
    return {"X-API-Key": key}


def test_missing_api_key_returns_401():
    resp = client.get("/api/v1/fixtures")
    assert resp.status_code == 401


def test_invalid_api_key_returns_401():
    resp = client.get("/api/v1/fixtures", headers=_auth("not-a-real-key"))
    assert resp.status_code == 401


def _seed_fixtures(season: str) -> tuple[int, int]:
    """Insert a near (1 day) and far (10 day) upcoming match, return their ids."""
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    with get_session() as session:
        home, away = session.query(Team).order_by(Team.id).limit(2).all()

        near = Match(
            season=season,
            kickoff_utc=now + dt.timedelta(days=1),
            home_team_id=home.id,
            away_team_id=away.id,
            status="SCHEDULED",
        )
        far = Match(
            season=season,
            kickoff_utc=now + dt.timedelta(days=10),
            home_team_id=away.id,
            away_team_id=home.id,
            status="SCHEDULED",
        )
        session.add_all([near, far])
        session.flush()
        return near.id, far.id


def test_free_tier_fixture_horizon_capping():
    near_id, far_id = _seed_fixtures("2099-test-free")

    # Free tier is capped to a 3-day horizon regardless of the requested days_ahead.
    resp = client.get("/api/v1/fixtures", params={"days_ahead": 14}, headers=_auth(FREE_KEY))
    assert resp.status_code == 200
    ids = {f["match_id"] for f in resp.json()}
    assert near_id in ids
    assert far_id not in ids


def test_pro_tier_sees_full_requested_horizon():
    near_id, far_id = _seed_fixtures("2099-test-pro")

    resp = client.get("/api/v1/fixtures", params={"days_ahead": 14}, headers=_auth(PRO_KEY))
    assert resp.status_code == 200
    ids = {f["match_id"] for f in resp.json()}
    assert near_id in ids
    assert far_id in ids


def test_free_tier_daily_quota_enforcement():
    with get_session() as session:
        record = session.query(ApiKey).filter_by(owner_label="demo-free").one()
        record.requests_today = record.daily_quota - 1
        record.quota_reset_at = dt.datetime.now(dt.UTC).replace(tzinfo=None) + dt.timedelta(days=1)

    # One request still within quota.
    resp = client.get("/api/v1/fixtures", headers=_auth(FREE_KEY))
    assert resp.status_code == 200

    # Quota now exhausted.
    resp = client.get("/api/v1/fixtures", headers=_auth(FREE_KEY))
    assert resp.status_code == 429

    with get_session() as session:
        record = session.query(ApiKey).filter_by(owner_label="demo-free").one()
        record.requests_today = 0


def test_accuracy_history_requires_pro_tier():
    resp = client.get("/api/v1/accuracy/history", headers=_auth(FREE_KEY))
    assert resp.status_code == 403


def test_accuracy_history_empty_for_pro_tier_with_no_active_model():
    resp = client.get("/api/v1/accuracy/history", headers=_auth(PRO_KEY))
    assert resp.status_code == 200
    assert resp.json() == []


def test_accuracy_summary_404_when_no_active_model():
    resp = client.get("/api/v1/accuracy/summary", headers=_auth(PRO_KEY))
    assert resp.status_code == 404


def test_accuracy_calibration_403_for_pro_tier():
    """Calibration data is an elite-only perk; pro tier is filtered+full-probability
    but does not get live calibration access."""
    resp = client.get("/api/v1/accuracy/calibration", headers=_auth(PRO_KEY))
    assert resp.status_code == 403


def test_accuracy_calibration_404_for_elite_tier_when_no_active_model():
    resp = client.get("/api/v1/accuracy/calibration", headers=_auth(ELITE_KEY))
    assert resp.status_code == 404


def test_accuracy_by_tier_404_when_no_active_model():
    resp = client.get("/api/v1/accuracy/by_tier", headers=_auth(FREE_KEY))
    assert resp.status_code == 404


def test_subscription_tiers_lists_free_pro_elite():
    resp = client.get("/api/v1/subscription/tiers", headers=_auth(FREE_KEY))
    assert resp.status_code == 200
    tiers = {t["tier"]: t for t in resp.json()}
    assert set(tiers) == {"free", "pro", "elite"}
    assert tiers["free"]["monthly_price_usd"] == 0
    assert tiers["pro"]["monthly_price_usd"] == 30
    assert tiers["elite"]["monthly_price_usd"] == 100


def test_subscription_me_reflects_caller_tier():
    resp = client.get("/api/v1/subscription/me", headers=_auth(PRO_KEY))
    assert resp.status_code == 200
    body = resp.json()
    assert body["tier"] == "pro"
    assert body["limits"]["min_confidence"] == 0.88
    assert body["limits"]["calibration_access"] is False


def test_free_tier_predictions_unfiltered_by_confidence():
    """Free tier shows every upcoming fixture's raw prediction (no active model
    here, so predicted_outcome is None, but the fixture itself is not filtered out)."""
    near_id, _ = _seed_fixtures("2099-test-free-unfiltered")

    resp = client.get("/api/v1/predictions/upcoming", params={"days_ahead": 1}, headers=_auth(FREE_KEY))
    assert resp.status_code == 200
    ids = {p["match_id"] for p in resp.json()}
    assert near_id in ids


def test_pro_tier_predictions_filtered_without_active_model():
    """Pro tier filters by confidence; with no active model, no prediction can
    clear the threshold, so the fixture is omitted entirely."""
    near_id, _ = _seed_fixtures("2099-test-pro-filtered")

    resp = client.get("/api/v1/predictions/upcoming", params={"days_ahead": 1}, headers=_auth(PRO_KEY))
    assert resp.status_code == 200
    ids = {p["match_id"] for p in resp.json()}
    assert near_id not in ids


# Held-out test-set predictions for a fake model version, used by the
# /accuracy/by_tier tests below. Three samples clear the 0.88 pro threshold
# (2/3 correct); all six clear the free tier's "no filter" (3/6 correct).
_BY_TIER_TEST_PREDICTIONS = [
    {"confidence": 0.95, "correct": True},
    {"confidence": 0.92, "correct": True},
    {"confidence": 0.90, "correct": False},
    {"confidence": 0.60, "correct": True},
    {"confidence": 0.55, "correct": False},
    {"confidence": 0.40, "correct": False},
]


def test_accuracy_by_tier_with_active_model():
    """Demonstrates the accuracy gap between tiers on the same held-out matches."""
    with get_session() as session:
        session.add(
            ModelVersion(
                name="test-model-by-tier",
                feature_names=["f1"],
                metrics={"test_predictions": _BY_TIER_TEST_PREDICTIONS},
                artifact_path="unused",
                is_active=True,
            )
        )

    resp = client.get("/api/v1/accuracy/by_tier", headers=_auth(FREE_KEY))
    assert resp.status_code == 200
    body = resp.json()
    assert body["test_size"] == 6

    free = body["tiers"]["free"]
    assert free["n_samples"] == 6
    assert free["accuracy"] == pytest.approx(3 / 6)

    pro = body["tiers"]["pro"]
    assert pro["min_confidence"] == pytest.approx(0.88)
    assert pro["n_samples"] == 3
    assert pro["accuracy"] == pytest.approx(2 / 3)

    elite_default = body["tiers"]["elite"]
    assert elite_default["n_samples"] == 3
    assert elite_default["accuracy"] == pytest.approx(2 / 3)


def test_accuracy_by_tier_elite_custom_threshold():
    resp = client.get(
        "/api/v1/accuracy/by_tier", params={"min_confidence": 0.5}, headers=_auth(ELITE_KEY)
    )
    assert resp.status_code == 200
    elite = resp.json()["tiers"]["elite"]
    # confidence > 0.5 -> the 0.95/0.92/0.90/0.60/0.55 samples (5 of 6), 3 correct
    assert elite["n_samples"] == 5
    assert elite["accuracy"] == pytest.approx(3 / 5)


def test_accuracy_by_tier_custom_threshold_ignored_for_non_elite():
    resp = client.get(
        "/api/v1/accuracy/by_tier", params={"min_confidence": 0.5}, headers=_auth(FREE_KEY)
    )
    assert resp.status_code == 200
    elite = resp.json()["tiers"]["elite"]
    # free tier cannot override the threshold -> elite bucket still uses 0.88
    assert elite["n_samples"] == 3
