"""Fixture listing endpoint."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query

from football_predictor.api.auth import AuthContext, get_auth_context
from football_predictor.api.schemas import FixtureOut
from football_predictor.db.models import Match, Team
from football_predictor.db.session import get_session

router = APIRouter()


@router.get("/fixtures", response_model=list[FixtureOut])
def list_fixtures(
    days_ahead: int = Query(7, ge=1, le=30, description="How many days ahead to look"),
    auth: AuthContext = Depends(get_auth_context),
) -> list[FixtureOut]:
    """Upcoming Premier League fixtures, capped by the caller's tier horizon."""
    horizon = min(days_ahead, auth.limits["fixture_horizon_days"])
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    cutoff = now + dt.timedelta(days=horizon)

    with get_session() as session:
        matches = (
            session.query(Match)
            .filter(Match.kickoff_utc >= now, Match.kickoff_utc <= cutoff)
            .order_by(Match.kickoff_utc)
            .all()
        )
        teams = {t.id: t.canonical_name for t in session.query(Team).all()}

        return [
            FixtureOut(
                match_id=m.id,
                season=m.season,
                kickoff_utc=m.kickoff_utc,
                home_team=teams[m.home_team_id],
                away_team=teams[m.away_team_id],
                status=m.status,
            )
            for m in matches
        ]
