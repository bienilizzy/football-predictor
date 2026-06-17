"""Tests for league/market filtering rules (no DB/model required)."""
from __future__ import annotations

from football_predictor.models.market_filters import MIN_MARKET_CONFIDENCE, filter_prediction


def test_turkish_super_lig_over_1_5_above_threshold_is_returned():
    result = filter_prediction("Turkish Super Lig", confidence=0.93)

    assert result is not None
    assert result["market"] == "Over 1.5 Goals"
    assert result["historical_hit_rate"] == 0.88


def test_turkish_super_lig_below_threshold_is_filtered():
    assert filter_prediction("Turkish Super Lig", confidence=0.85) is None


def test_confidence_exactly_at_threshold_is_filtered():
    """The gate is a strict `>` so exactly MIN_MARKET_CONFIDENCE is withheld."""
    assert filter_prediction("Turkish Super Lig", confidence=MIN_MARKET_CONFIDENCE) is None


def test_serie_a_btts_no_requires_form_gap_and_confidence():
    result = filter_prediction("Serie A", confidence=0.95, form_difference=0.7)

    assert result is not None
    assert result["market"] == "Both Teams to Score - No"
    assert result["historical_hit_rate"] == 0.84


def test_serie_a_btts_no_filtered_when_form_gap_too_small():
    assert filter_prediction("Serie A", confidence=0.95, form_difference=0.4) is None


def test_serie_a_btts_no_filtered_when_confidence_too_low_despite_form_gap():
    assert filter_prediction("Serie A", confidence=0.85, form_difference=0.7) is None


def test_serie_a_form_gap_is_checked_by_absolute_value():
    result = filter_prediction("Serie A", confidence=0.95, form_difference=-0.7)

    assert result is not None


def test_premier_league_never_returns_a_prediction():
    assert filter_prediction("Premier League", confidence=0.99) is None


def test_unknown_league_returns_none():
    assert filter_prediction("Bundesliga", confidence=0.99) is None
