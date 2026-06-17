# Football Predictor

A self-contained multi-sport prediction system: historical Premier League data
ingestion, no-leakage feature engineering, a calibrated XGBoost model, a
five-analyst LLM committee (Claude), a tiered FastAPI backend, and a Streamlit
dashboard for predictions and tracked accuracy.

## Architecture

```
football-predictor/
├── config/settings.py          # pydantic-settings: env vars, paths, league/season config
├── data/                        # raw CSV cache + sqlite db (gitignored)
├── models/                      # saved model artifacts (gitignored)
├── src/football_predictor/
│   ├── db/                      # SQLAlchemy models, session, init/seed script
│   ├── reference_data.py        # 20 PL clubs mapped across all data sources + stadium coords
│   ├── ingestion/                # football-data.co.uk / .org, Understat, Open-Meteo clients
│   ├── features/                 # form, xG, H2H, referee bias, weather feature builders
│   ├── models/                   # train / predict / evaluation / accuracy / sport resolution
│   ├── sports/                   # multi-sport data layer (Sportmonks) + per-sport features
│   ├── agents/                   # LLM analyst committee (Claude) + Redis cache
│   ├── api/                      # FastAPI app, tiered auth, routers
│   └── dashboard/                # Streamlit app
├── scripts/                      # CLI entry points for each pipeline stage
└── tests/                        # pytest suite
```

## Data sources

| Source | Used for | Auth |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/) (`E0.csv` per season) | Historical results, referees, cards, shots — training backbone | None |
| [football-data.org](https://www.football-data.org/) v4 API | Upcoming PL fixture schedule; result resolution for sport predictions | Free API key |
| [Understat](https://understat.com/) (via `understatapi`) | xG / xGA per match, 2014-15 onward | None (scrapes embedded JSON) |
| [Open-Meteo](https://open-meteo.com/) archive + forecast APIs | Weather (temp/precip/wind) at kickoff, by stadium location | None |
| [Sportmonks](https://www.sportmonks.com/) v3 API | Upcoming fixtures and results for football, cricket, tennis, F1 | Free API key |
| [Anthropic Claude](https://console.anthropic.com/) | LLM analyst committee — five-persona independent outcome estimates | API key |

## Setup

1. Create a virtualenv and install the project:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -e .
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in the required keys:

   | Variable | Required for | Where to get it |
   |---|---|---|
   | `FOOTBALL_DATA_ORG_API_KEY` | Upcoming PL fixtures + sport result resolution | https://www.football-data.org/client/register |
   | `SPORTMONKS_API_KEY` | Multi-sport predictions (cricket / tennis / F1) | https://www.sportmonks.com/ |
   | `ANTHROPIC_API_KEY` | LLM analyst committee (`/predictions/llm`, `/sports/{sport}/upcoming`) | https://console.anthropic.com/ |

   Keys are only required for the features that use them; historical backfill
   and model training work without any of them.

3. (Optional) Start Redis for LLM committee response caching. Without it every
   repeat prediction re-queries the committee, but the API still works:

   ```bash
   docker compose up -d redis
   ```

4. Initialize the database (creates tables, seeds the 20 current PL clubs and
   three demo API keys — one per tier):

   ```bash
   python scripts/init_db.py
   ```

## Running the pipeline

Run these in order to go from an empty database to a trained model with predictions:

```bash
# 1. Backfill historical seasons (results, referees, cards, xG, weather).
#    Seasons come from HISTORICAL_SEASONS in .env (default: 2223,2324,2425).
python scripts/ingest_historical.py

# 2. Build the engineered feature matrix for every match.
python scripts/build_features.py

# 3. Train an XGBoost model on a chronological train/calibrate/test split,
#    calibrate its probabilities, evaluate, and activate it as the current model.
python scripts/train_model.py

# 4. Score upcoming/unresolved PL fixtures with the active model.
python scripts/generate_predictions.py
```

For ongoing use:

```bash
# Daily refresh: current-season results/xG, upcoming fixtures, and weather.
python scripts/ingest_latest.py

# Rebuild features and predictions after new data arrives.
python scripts/build_features.py && python scripts/generate_predictions.py

# Score predictions whose matches have since finished (PL model).
python scripts/update_accuracy.py

# Resolve actual outcomes for multi-sport predictions (cricket/tennis/F1/football
# via Sportmonks). Run after matches have finished.
python scripts/resolve_sport_predictions.py
```

## Running the API and dashboard

```bash
# Terminal 1: API
uvicorn football_predictor.api.main:app --reload

# Terminal 2: dashboard (uses the seeded "pro" demo key by default)
streamlit run src/football_predictor/dashboard/app.py
```

The dashboard reads `FOOTBALL_PREDICTOR_API_URL` (default `http://localhost:8000`)
and `FOOTBALL_PREDICTOR_API_KEY` (default `demo-pro-key`) from the environment.
Use `FOOTBALL_PREDICTOR_API_KEY=demo-elite-key` to see LLM committee agent
breakdowns and calibration curves.

### API tiers

Auth is via the `X-API-Key` header. `scripts/init_db.py` seeds one demo key per
tier:

| Tier | Demo key | Daily quota | Fixture horizon | Probabilities | Feature contributions | LLM committee | Accuracy history | Calibration |
|---|---|---|---|---|---|---|---|---|
| free | `demo-free-key` | 50 | 3 days | predicted outcome only | no | no | no | no |
| pro | `demo-pro-key` | 500 | 14 days | full H/D/A probabilities | yes (SHAP-style) | no | yes | no |
| elite | `demo-elite-key` | 5000 | 30 days | full H/D/A probabilities | yes (SHAP-style) | yes | yes | yes |

Elite tier additionally allows a caller-defined `min_confidence` threshold on
prediction endpoints, overriding the system default.

### Endpoints

**PL football (XGBoost model)**
- `GET /api/v1/fixtures?days_ahead=N` — upcoming PL fixtures (horizon capped per tier)
- `GET /api/v1/predictions/upcoming?days_ahead=N[&min_confidence=F]` — predictions for upcoming fixtures
- `GET /api/v1/predictions/{match_id}[?min_confidence=F]` — single-match prediction (+ feature contributions for pro/elite)
- `POST /api/v1/predictions/llm` body `{"match_id": N}` — LLM committee prediction (elite tier only)

**Accuracy / calibration**
- `GET /api/v1/accuracy/summary` — accuracy/log-loss/Brier for the active model
- `GET /api/v1/accuracy/calibration` — reliability-curve data from the held-out test set (elite tier)
- `GET /api/v1/accuracy/history` — per-match predicted vs. actual outcomes (pro/elite)
- `GET /api/v1/accuracy/by_tier` — confidence-bucketed accuracy on the held-out test set

**Multi-sport (LLM committee)**
- `GET /api/v1/sports/{sport}/upcoming?days_ahead=N` — LLM committee predictions for football/cricket/tennis/f1 (pro/elite)
- `GET /api/v1/sports/leaderboard` — resolved-prediction accuracy per sport, bucketed by tier
- `GET /api/v1/sports/{sport}/calibration` — reliability curve for a sport's LLM predictions (elite tier)

**Subscription**
- `GET /api/v1/subscription/tiers` — public catalog of subscription tiers and capabilities
- `GET /api/v1/subscription/me` — the caller's current tier, capabilities, and quota usage

## LLM analyst committee

The elite-tier `/predictions/llm` endpoint and all `/sports/{sport}/upcoming`
endpoints run a committee of five independent Claude analyst personas in parallel:

| Agent | Expertise |
|---|---|
| Form Analyst | Recent results, scoring trends, momentum |
| Tactical Analyst | Playing styles, matchups, formations |
| Context Analyst | Injuries, fixture congestion, motivation, stakes |
| Market Analyst | Betting market signals and implied probabilities |
| Historical Pattern Analyst | Head-to-head history and long-run patterns |

If the agents broadly agree (per-class variance below a threshold), their averaged
probabilities are returned as a high-confidence committee prediction. Otherwise the
system falls back to the trained XGBoost model for PL football fixtures with stored
features, or returns the committee average flagged as low-confidence for other sports.

Responses are cached in Redis for 1 hour per sport/fixture to avoid redundant API
calls. The cache is optional — if Redis is unreachable every request goes to Claude
directly.

## Testing

```bash
pytest
```

Tests cover:
- Feature math (rolling form/xG, head-to-head, referee bias) against hand-computed examples
- Calibration wrapper output and evaluation metrics
- API auth, tier gating, fixture-horizon capping, and daily quota enforcement
- LLM committee: mocked Anthropic responses, variance threshold, caching, tier gate

## Known limitations

- **Referee assignments for upcoming fixtures** are generally not available from
  free sources until ~24-48h before kickoff. Referee features for unplayed matches
  fall back to league-average stats (`referee_known=0`).
- **Understat scraping** (`understatapi`) depends on the site's embedded JSON
  structure and may break if Understat changes its page format.
- **Team reference table** (`reference_data.py`) covers the 2025-26 Premier League's
  20 clubs (plus a few recently-relegated clubs for historical H2H/form coverage)
  and needs manual updates after each promotion/relegation cycle.
- **SQLite over network filesystems**: `DATABASE_URL` defaults to a `?uri=true&nolock=1`
  SQLite URI with `journal_mode=MEMORY`, required when the project lives on a
  network-mounted path (e.g. `\\wsl.localhost` from Windows). This trades away
  SQLite's concurrent-writer locking guarantees — fine for a single-process local
  setup, but swap in a Postgres `DATABASE_URL` for any multi-writer/production
  deployment.
- **`requests` vs `httpx`**: on some Python/OpenSSL builds, `requests`/urllib3's TLS
  handshake fails (`SSLEOFError`) against certain hosts. The football-data.co.uk
  client uses `httpx` instead. If you see similar SSL errors from other clients,
  the same swap should fix it.
- **Sport prediction resolution** for cricket, tennis, and F1 depends on the
  Sportmonks API returning structured score/result data in the expected format.
  If Sportmonks changes its response shape, `scripts/resolve_sport_predictions.py`
  may need updating.
- **LLM committee accuracy** for non-football sports reflects general world
  knowledge rather than a trained statistical model and should be treated
  accordingly. Football predictions backed by XGBoost (low committee variance)
  are more reliable than LLM-only estimates.
