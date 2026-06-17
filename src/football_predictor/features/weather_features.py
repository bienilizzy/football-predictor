"""Weather features: kickoff temperature/precipitation/wind, with safe defaults."""
from __future__ import annotations

import pandas as pd

# Roughly typical conditions for a UK matchday, used when no weather record exists.
DEFAULTS = {"weather_temp_c": 12.0, "weather_precip_mm": 0.0, "weather_wind_kph": 10.0}

WEATHER_COLUMNS = list(DEFAULTS.keys())


def compute_weather_features(matches: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """`matches` needs `match_id`. `weather` needs match_id, temp_c, precip_mm, wind_kph."""
    if weather.empty:
        merged = matches[["match_id"]].copy()
        merged["temp_c"] = pd.NA
        merged["precip_mm"] = pd.NA
        merged["wind_kph"] = pd.NA
    else:
        merged = matches[["match_id"]].merge(
            weather[["match_id", "temp_c", "precip_mm", "wind_kph"]], on="match_id", how="left"
        )

    out = pd.DataFrame(
        {
            "match_id": merged["match_id"],
            "weather_temp_c": merged["temp_c"],
            "weather_precip_mm": merged["precip_mm"],
            "weather_wind_kph": merged["wind_kph"],
        }
    )
    for col, default in DEFAULTS.items():
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
    return out
