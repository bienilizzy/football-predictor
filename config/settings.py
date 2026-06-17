"""Centralized application configuration, loaded from environment variables / .env."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- External APIs ---
    football_data_org_api_key: str = ""
    sportmonks_api_key: str = ""
    sportmonks_tier: str = "basic"
    sportscore_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # --- Caching ---
    # Used by the LLM analyst committee's response cache
    # (src/football_predictor/agents/cache.py).
    redis_url: str = "redis://localhost:6379/0"

    # --- Storage ---
    # `?uri=true&nolock=1` + journal_mode=MEMORY (set in db/session.py) is required
    # for SQLite to work on network-mounted filesystems (e.g. WSL's \\wsl.localhost
    # UNC paths accessed from Windows), where POSIX file locking is unreliable.
    database_url: str = "sqlite:///file:./data/football_predictor.db?uri=true&nolock=1"
    data_raw_dir: Path = Path("./data/raw")
    model_dir: Path = Path("./models")

    # --- League identifiers across data sources ---
    fd_org_competition_code: str = "PL"
    fd_co_uk_league_code: str = "E0"
    understat_league: str = "EPL"

    # --- Seasons ---
    historical_seasons: str = "2223,2324,2425"
    current_season: str = "2025-2026"

    @property
    def historical_season_codes(self) -> list[str]:
        """football-data.co.uk season codes, e.g. ['2223', '2324', '2425']."""
        return [s.strip() for s in self.historical_seasons.split(",") if s.strip()]

    @property
    def db_file_path(self) -> Path:
        """Filesystem path of the SQLite database file, parsed out of `database_url`."""
        raw = self.database_url.removeprefix("sqlite:///")
        if raw.startswith("file:"):
            raw = raw[len("file:"):]
        return Path(raw.split("?", 1)[0])


settings = Settings()

# Ensure local storage directories exist.
settings.data_raw_dir.mkdir(parents=True, exist_ok=True)
settings.model_dir.mkdir(parents=True, exist_ok=True)
settings.db_file_path.parent.mkdir(parents=True, exist_ok=True)
