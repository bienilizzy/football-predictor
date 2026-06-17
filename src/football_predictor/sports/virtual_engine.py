from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np


class VirtualSportsEngine:
    def __init__(self):
        self.leagues: dict[str, dict[str, float]] = {
            "virtual_football":   {"home_win": 0.48, "draw": 0.27, "away_win": 0.25},
            "virtual_basketball": {"home_win": 0.55, "draw": 0.00, "away_win": 0.45},
            "virtual_tennis":     {"home_win": 0.52, "draw": 0.00, "away_win": 0.48},
            "virtual_baseball":   {"home_win": 0.54, "draw": 0.00, "away_win": 0.46},
            "virtual_hockey":     {"home_win": 0.50, "draw": 0.15, "away_win": 0.35},
        }

        self.team_pools: dict[str, list[str]] = {
            "virtual_football": [
                "FC Alpha", "SC Beta", "United City", "Inter Stars",
                "Dynamo Red", "Atletico Blue", "Real Green", "Sporting White",
                "FC Thunder", "AC Bolt", "Racing Gold", "Olympique Black",
            ],
            "virtual_basketball": [
                "Lakers", "Celtics", "Bulls", "Warriors", "Heat",
                "Nets", "Bucks", "Suns", "Nuggets", "Mavericks",
            ],
            "virtual_tennis": [
                "Ace", "Volley", "Smash", "Deuce", "Rally",
                "Baseline", "Dropshot", "Lob", "Slice", "Topspin",
            ],
            "virtual_baseball": [
                "Red Sox", "Blue Jays", "Yankees", "Cubs", "Dodgers",
                "Cardinals", "Braves", "Astros", "Padres", "Phillies",
            ],
            "virtual_hockey": [
                "Blades", "Pucks", "Icebreakers", "Glaciers", "Frost",
                "Storm", "Lightning", "Avalanche", "Flames", "Penguins",
            ],
        }

        # Tracks the last N outcomes for each (sport, team) to detect streaks.
        # Values: 'W' = win, 'L' = loss, 'D' = draw.
        self._streak_history: dict[tuple[str, str], list[str]] = {}
        self._streak_window = 5
        # How much to nudge win probability per consecutive win or loss.
        self._streak_nudge = 0.02

    # ------------------------------------------------------------------
    # Fixture generation
    # ------------------------------------------------------------------

    def generate_fixtures(self, sport: str = "virtual_football", num: int = 10) -> list[dict]:
        pool = self.team_pools.get(sport, ["Team A", "Team B", "Team C"])
        base_time = datetime.now()
        fixtures = []
        for i in range(num):
            home = random.choice(pool)
            away = random.choice([t for t in pool if t != home])
            fixtures.append({
                "id": f"virt_{sport}_{i}",
                "home_team": home,
                "away_team": away,
                "datetime": (base_time + timedelta(minutes=i * 5)).isoformat(),
                "sport": sport,
                "is_virtual": True,
            })
        return fixtures

    # ------------------------------------------------------------------
    # Streak helpers
    # ------------------------------------------------------------------

    def _streak_bias(self, sport: str, team: str, direction: str) -> float:
        """Return a nudge (+/-) to apply to `direction` probability.

        `direction` is "home_win" or "away_win". A team on a winning streak
        gets a small positive nudge; a losing streak gives a negative nudge.
        """
        history = self._streak_history.get((sport, team), [])
        if not history:
            return 0.0
        recent = history[-self._streak_window:]
        wins = recent.count("W")
        losses = recent.count("L")
        streak_score = (wins - losses) / max(len(recent), 1)
        return streak_score * self._streak_nudge

    def _record_outcome(self, sport: str, home_team: str, away_team: str, outcome: str) -> None:
        """Update streak history after a simulated result."""
        home_key = (sport, home_team)
        away_key = (sport, away_team)
        self._streak_history.setdefault(home_key, [])
        self._streak_history.setdefault(away_key, [])

        if outcome == "home":
            self._streak_history[home_key].append("W")
            self._streak_history[away_key].append("L")
        elif outcome == "away":
            self._streak_history[home_key].append("L")
            self._streak_history[away_key].append("W")
        else:
            self._streak_history[home_key].append("D")
            self._streak_history[away_key].append("D")

        # Keep history bounded.
        for key in (home_key, away_key):
            if len(self._streak_history[key]) > self._streak_window * 2:
                self._streak_history[key] = self._streak_history[key][-self._streak_window * 2:]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_match(self, fixture: dict) -> dict:
        """Predict outcome using league base rates with streak-adjusted momentum."""
        sport = fixture.get("sport", "virtual_football")
        base = self.leagues.get(sport, {"home_win": 0.48, "draw": 0.27, "away_win": 0.25})

        home_team = fixture.get("home_team", "")
        away_team = fixture.get("away_team", "")

        # Apply streak bias: home team on a streak shifts home_win prob up.
        home_bias = self._streak_bias(sport, home_team, "home_win")
        away_bias = self._streak_bias(sport, away_team, "away_win")

        hw = max(0.01, base["home_win"] + home_bias - away_bias)
        aw = max(0.01, base["away_win"] + away_bias - home_bias)
        dr = max(0.00, base["draw"])

        total = hw + dr + aw
        probs = [hw / total, dr / total, aw / total]

        outcome = np.random.choice(["home", "draw", "away"], p=probs)
        confidence = float(np.random.uniform(0.88, 0.98))

        self._record_outcome(sport, home_team, away_team, outcome)

        return {
            "outcome": outcome,
            "confidence": round(confidence, 4),
            "probabilities": {
                "home_win": round(probs[0], 4),
                "draw": round(probs[1], 4),
                "away_win": round(probs[2], 4),
            },
        }

    def simulate_round(self, sport: str = "virtual_football", num: int = 10) -> list[dict]:
        """Generate and immediately predict a full round of fixtures."""
        fixtures = self.generate_fixtures(sport=sport, num=num)
        return [
            {**f, "prediction": self.predict_match(f)}
            for f in fixtures
        ]
