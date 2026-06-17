"""Static reference data for the Premier League.

Maps each club's name across the three data sources used by this project, plus
stadium coordinates used for weather lookups.

NOTE: The 20 clubs marked ``is_current=True`` reflect the 2025-2026 Premier League
season. After each promotion/relegation cycle this list needs to be updated. A
handful of recently-relegated clubs are also included with ``is_current=False`` so
that historical seasons used for training have complete rolling-form/H2H history
for every team they faced.
"""
from __future__ import annotations

PREMIER_LEAGUE_TEAMS: list[dict] = [
    {
        "canonical_name": "Arsenal",
        "fd_org_name": "Arsenal FC",
        "fd_co_uk_name": "Arsenal",
        "understat_name": "Arsenal",
        "stadium": "Emirates Stadium",
        "latitude": 51.5549,
        "longitude": -0.1084,
        "is_current": True,
    },
    {
        "canonical_name": "Aston Villa",
        "fd_org_name": "Aston Villa FC",
        "fd_co_uk_name": "Aston Villa",
        "understat_name": "Aston Villa",
        "stadium": "Villa Park",
        "latitude": 52.5092,
        "longitude": -1.8848,
        "is_current": True,
    },
    {
        "canonical_name": "Bournemouth",
        "fd_org_name": "AFC Bournemouth",
        "fd_co_uk_name": "Bournemouth",
        "understat_name": "Bournemouth",
        "stadium": "Vitality Stadium",
        "latitude": 50.7352,
        "longitude": -1.8380,
        "is_current": True,
    },
    {
        "canonical_name": "Brentford",
        "fd_org_name": "Brentford FC",
        "fd_co_uk_name": "Brentford",
        "understat_name": "Brentford",
        "stadium": "Gtech Community Stadium",
        "latitude": 51.4906,
        "longitude": -0.2885,
        "is_current": True,
    },
    {
        "canonical_name": "Brighton",
        "fd_org_name": "Brighton & Hove Albion FC",
        "fd_co_uk_name": "Brighton",
        "understat_name": "Brighton",
        "stadium": "Falmer Stadium (Amex)",
        "latitude": 50.8617,
        "longitude": -0.0837,
        "is_current": True,
    },
    {
        "canonical_name": "Burnley",
        "fd_org_name": "Burnley FC",
        "fd_co_uk_name": "Burnley",
        "understat_name": "Burnley",
        "stadium": "Turf Moor",
        "latitude": 53.7890,
        "longitude": -2.2308,
        "is_current": True,
    },
    {
        "canonical_name": "Chelsea",
        "fd_org_name": "Chelsea FC",
        "fd_co_uk_name": "Chelsea",
        "understat_name": "Chelsea",
        "stadium": "Stamford Bridge",
        "latitude": 51.4817,
        "longitude": -0.1910,
        "is_current": True,
    },
    {
        "canonical_name": "Crystal Palace",
        "fd_org_name": "Crystal Palace FC",
        "fd_co_uk_name": "Crystal Palace",
        "understat_name": "Crystal Palace",
        "stadium": "Selhurst Park",
        "latitude": 51.3983,
        "longitude": -0.0856,
        "is_current": True,
    },
    {
        "canonical_name": "Everton",
        "fd_org_name": "Everton FC",
        "fd_co_uk_name": "Everton",
        "understat_name": "Everton",
        "stadium": "Hill Dickinson Stadium",
        "latitude": 53.4500,
        "longitude": -2.9925,
        "is_current": True,
    },
    {
        "canonical_name": "Fulham",
        "fd_org_name": "Fulham FC",
        "fd_co_uk_name": "Fulham",
        "understat_name": "Fulham",
        "stadium": "Craven Cottage",
        "latitude": 51.4749,
        "longitude": -0.2217,
        "is_current": True,
    },
    {
        "canonical_name": "Leeds United",
        "fd_org_name": "Leeds United FC",
        "fd_co_uk_name": "Leeds",
        "understat_name": "Leeds",
        "stadium": "Elland Road",
        "latitude": 53.7778,
        "longitude": -1.5722,
        "is_current": True,
    },
    {
        "canonical_name": "Liverpool",
        "fd_org_name": "Liverpool FC",
        "fd_co_uk_name": "Liverpool",
        "understat_name": "Liverpool",
        "stadium": "Anfield",
        "latitude": 53.4308,
        "longitude": -2.9608,
        "is_current": True,
    },
    {
        "canonical_name": "Manchester City",
        "fd_org_name": "Manchester City FC",
        "fd_co_uk_name": "Man City",
        "understat_name": "Manchester City",
        "stadium": "Etihad Stadium",
        "latitude": 53.4831,
        "longitude": -2.2004,
        "is_current": True,
    },
    {
        "canonical_name": "Manchester United",
        "fd_org_name": "Manchester United FC",
        "fd_co_uk_name": "Man United",
        "understat_name": "Manchester United",
        "stadium": "Old Trafford",
        "latitude": 53.4631,
        "longitude": -2.2913,
        "is_current": True,
    },
    {
        "canonical_name": "Newcastle United",
        "fd_org_name": "Newcastle United FC",
        "fd_co_uk_name": "Newcastle",
        "understat_name": "Newcastle United",
        "stadium": "St James' Park",
        "latitude": 54.9756,
        "longitude": -1.6217,
        "is_current": True,
    },
    {
        "canonical_name": "Nottingham Forest",
        "fd_org_name": "Nottingham Forest FC",
        "fd_co_uk_name": "Nott'm Forest",
        "understat_name": "Nottingham Forest",
        "stadium": "City Ground",
        "latitude": 52.9400,
        "longitude": -1.1328,
        "is_current": True,
    },
    {
        "canonical_name": "Sunderland",
        "fd_org_name": "Sunderland AFC",
        "fd_co_uk_name": "Sunderland",
        "understat_name": "Sunderland",
        "stadium": "Stadium of Light",
        "latitude": 54.9144,
        "longitude": -1.3883,
        "is_current": True,
    },
    {
        "canonical_name": "Tottenham",
        "fd_org_name": "Tottenham Hotspur FC",
        "fd_co_uk_name": "Tottenham",
        "understat_name": "Tottenham",
        "stadium": "Tottenham Hotspur Stadium",
        "latitude": 51.6043,
        "longitude": -0.0664,
        "is_current": True,
    },
    {
        "canonical_name": "West Ham",
        "fd_org_name": "West Ham United FC",
        "fd_co_uk_name": "West Ham",
        "understat_name": "West Ham",
        "stadium": "London Stadium",
        "latitude": 51.5386,
        "longitude": -0.0166,
        "is_current": True,
    },
    {
        "canonical_name": "Wolverhampton Wanderers",
        "fd_org_name": "Wolverhampton Wanderers FC",
        "fd_co_uk_name": "Wolves",
        "understat_name": "Wolverhampton Wanderers",
        "stadium": "Molineux Stadium",
        "latitude": 52.5903,
        "longitude": -2.1300,
        "is_current": True,
    },
    # --- Recently relegated clubs, kept for historical training data only ---
    {
        "canonical_name": "Leicester City",
        "fd_org_name": "Leicester City FC",
        "fd_co_uk_name": "Leicester",
        "understat_name": "Leicester",
        "stadium": "King Power Stadium",
        "latitude": 52.6204,
        "longitude": -1.1422,
        "is_current": False,
    },
    {
        "canonical_name": "Southampton",
        "fd_org_name": "Southampton FC",
        "fd_co_uk_name": "Southampton",
        "understat_name": "Southampton",
        "stadium": "St Mary's Stadium",
        "latitude": 50.9058,
        "longitude": -1.3911,
        "is_current": False,
    },
    {
        "canonical_name": "Sheffield United",
        "fd_org_name": "Sheffield United FC",
        "fd_co_uk_name": "Sheffield United",
        "understat_name": "Sheffield United",
        "stadium": "Bramall Lane",
        "latitude": 53.3703,
        "longitude": -1.4709,
        "is_current": False,
    },
    {
        "canonical_name": "Luton Town",
        "fd_org_name": "Luton Town FC",
        "fd_co_uk_name": "Luton",
        "understat_name": "Luton",
        "stadium": "Kenilworth Road",
        "latitude": 51.8844,
        "longitude": -0.4319,
        "is_current": False,
    },
    {
        "canonical_name": "Ipswich Town",
        "fd_org_name": "Ipswich Town FC",
        "fd_co_uk_name": "Ipswich",
        "understat_name": "Ipswich",
        "stadium": "Portman Road",
        "latitude": 52.0552,
        "longitude": 1.1450,
        "is_current": False,
    },
]


def fd_org_name_map() -> dict[str, str]:
    """football-data.org team name -> canonical name."""
    return {t["fd_org_name"]: t["canonical_name"] for t in PREMIER_LEAGUE_TEAMS}


def fd_co_uk_name_map() -> dict[str, str]:
    """football-data.co.uk team name -> canonical name."""
    return {t["fd_co_uk_name"]: t["canonical_name"] for t in PREMIER_LEAGUE_TEAMS}


def understat_name_map() -> dict[str, str]:
    """Understat team name -> canonical name."""
    return {t["understat_name"]: t["canonical_name"] for t in PREMIER_LEAGUE_TEAMS}
