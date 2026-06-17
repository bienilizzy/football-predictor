from __future__ import annotations

import requests
from datetime import datetime, timedelta

from config.settings import settings


class SportscoreClient:
    BASE_URL = "https://sportscore1.p.rapidapi.com"
    HOST = "sportscore1.p.rapidapi.com"

    def __init__(self, api_key: str | None = None):
        key = api_key if api_key is not None else settings.sportscore_api_key
        self.headers = {
            "x-rapidapi-host": self.HOST,
            "x-rapidapi-key": key,
        }

    def get_sports(self) -> dict:
        """Return mapping of sport names to IDs."""
        url = f"{self.BASE_URL}/sports"
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_fixtures(
        self,
        sport_id: int,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> dict:
        if not date_from:
            date_from = datetime.now().strftime("%Y-%m-%d")
        if not date_to:
            date_to = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        url = f"{self.BASE_URL}/events"
        params = {
            "sport_id": sport_id,
            "date_from": date_from,
            "date_to": date_to,
            "page": 1,
            "per_page": limit,
        }
        resp = requests.get(url, headers=self.headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_live_events(self, sport_id: int) -> dict:
        """In-play fixtures — to be implemented."""
        pass


# Quick test
if __name__ == "__main__":
    client = SportscoreClient()
    sports = client.get_sports()
    print(sports)
    # Test basketball
    bball = client.get_fixtures(2)
    print(bball)
