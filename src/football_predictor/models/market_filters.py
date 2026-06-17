"""League-specific betting market filters.

The trained H/D/A model in `predict.py` only covers the Premier League, where
its accuracy (~55-58%) is too low to be useful for market betting. This module
instead encodes per-league/market rules for *other* markets, based on
externally-sourced historical hit-rate research (provided by the project
owner, not computed by this codebase):

- Turkish Super Lig: "Over 1.5 Goals" has an ~88% historical hit rate.
- Serie A: "Both Teams to Score - No" has an ~84% historical hit rate, but
  only when the pre-match form difference between the two sides exceeds 0.6.
- Premier League: no market is surfaced at all.

On top of any league-specific condition, a prediction is only ever returned
if its market-specific confidence exceeds `MIN_MARKET_CONFIDENCE`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

MIN_MARKET_CONFIDENCE = 0.90


@dataclass(frozen=True)
class MarketRule:
    """A single league's recommended market and the conditions for surfacing it."""

    market: str
    historical_hit_rate: float
    extra_condition: Callable[[dict], bool] = field(default=lambda context: True)


def _serie_a_form_gap_exceeds_threshold(context: dict) -> bool:
    return abs(context.get("form_difference", 0.0)) > 0.6


# A `None` value means the league has no market reliable enough to surface.
LEAGUE_MARKET_RULES: dict[str, MarketRule | None] = {
    "Turkish Super Lig": MarketRule(
        market="Over 1.5 Goals",
        historical_hit_rate=0.88,
    ),
    "Serie A": MarketRule(
        market="Both Teams to Score - No",
        historical_hit_rate=0.84,
        extra_condition=_serie_a_form_gap_exceeds_threshold,
    ),
    "Premier League": None,
}


def filter_prediction(league: str, confidence: float, **context: object) -> dict | None:
    """Return the recommended market bet for `league`, or None if it's filtered out.

    `confidence` is the model's market-specific confidence for this match
    (e.g. P(Over 1.5 Goals) for Turkish Super Lig, P(BTTS = No) for Serie A).
    `context` carries any additional signals a rule's `extra_condition` needs,
    e.g. `form_difference=home_form_pts - away_form_pts` for Serie A.

    Returns None if the league has no market rule, the confidence does not
    exceed `MIN_MARKET_CONFIDENCE`, or the rule's extra condition fails.
    """
    rule = LEAGUE_MARKET_RULES.get(league)
    if rule is None:
        return None

    if confidence <= MIN_MARKET_CONFIDENCE:
        return None

    if not rule.extra_condition(context):
        return None

    return {
        "league": league,
        "market": rule.market,
        "confidence": confidence,
        "historical_hit_rate": rule.historical_hit_rate,
    }
