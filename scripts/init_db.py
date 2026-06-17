#!/usr/bin/env python
"""Create all tables and seed reference data / demo API keys."""
from football_predictor.db.init_db import DEMO_API_KEYS, init_db

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
    print("Demo API keys (for local dev / dashboard use):")
    for tier, (raw_key, quota) in DEMO_API_KEYS.items():
        print(f"  {tier:>5}: {raw_key}  (quota={quota}/day)")
