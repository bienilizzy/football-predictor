"""Streamlit dashboard for the Football Predictor API.

Run with: streamlit run src/football_predictor/dashboard/app.py
(requires the FastAPI backend to be running, e.g. `uvicorn football_predictor.api.main:app`)
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import random
import streamlit as st

def __render_predictor_cards(predictions, sport, show_booking=True, bookmaker="SportyBet"):
    """Display a list of predictions as cards."""
    if not predictions:
        st.info("No predictions available.")
        return

    # Map bookmaker to endpoint
    bookmaker_endpoint_map = {
        "SportyBet": "/api/v1/bookmaker/sportybet/convert",
        "Bet9ja": "/api/v1/bookmaker/bet9ja/convert",
        "1xBet": "/api/v1/bookmaker/onexbet/convert",
    }
    endpoint = bookmaker_endpoint_map.get(bookmaker, "/api/v1/bookmaker/sportybet/convert")

    for idx, pred in enumerate(predictions):
        with st.container():
            # Extract basic data
            participants = pred.get("participants", {})
            home = participants.get("home_team", "Home")
            away = participants.get("away_team", "Away")
            outcome = pred.get("predicted_outcome", "H")
            confidence = pred.get("confidence", 0.0)
            if confidence is None:
                confidence = 0.0

            # Derive probabilities from outcome and confidence
            if outcome == "H":
                p_home = confidence
                p_draw = (1 - confidence) * 0.5
                p_away = (1 - confidence) * 0.5
            elif outcome == "A":
                p_away = confidence
                p_home = (1 - confidence) * 0.5
                p_draw = (1 - confidence) * 0.5
            else:  # Draw
                p_draw = confidence
                p_home = (1 - confidence) * 0.5
                p_away = (1 - confidence) * 0.5

            # Clamp to avoid tiny negative values
            p_home = max(0.0, min(1.0, p_home))
            p_draw = max(0.0, min(1.0, p_draw))
            p_away = max(0.0, min(1.0, p_away))

            kickoff = pred.get("kickoff_utc", "TBD")
            external_id = pred.get("external_id", f"match_{idx}")

            # Display card
            st.subheader(f"{home} vs {away}")
            st.caption(f" {kickoff}")
            col1, col2, col3 = st.columns(3)
            col1.metric(" Home", f"{p_home:.0%}")
            col2.metric(" Draw", f"{p_draw:.0%}")
            col3.metric("Away", f"{p_away:.0%}")

            outcome_label = {"H": "Home win", "D": "Draw", "A": "Away win"}.get(outcome, "Unknown")
            st.success(f"**Prediction:** {outcome_label} (confidence {confidence:.0%})")

            # Booking code button
            if show_booking:
                if st.button(f"Get {bookmaker} Booking Code", key=f"code_{external_id}"):
                    payload = {
                        "match_id": external_id,
                        "prediction": outcome,
                        "odds": 2.10,
                        "sport": sport.lower()
                    }
                    headers = {"X-API-Key": API_KEY}
                    try:
                        resp = requests.post(
                            f"{API_BASE_URL}{endpoint}",
                            json=payload,
                            headers=headers,
                            timeout=10
                        )
                        if resp.status_code == 200:
                            code = resp.json().get("booking_code")
                            st.success(f" {bookmaker} Booking Code: **{code}**")
                        else:
                            st.error(f"Failed: {resp.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")
            st.divider() 

def _secrets_get(key: str, default: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return default


API_BASE_URL = os.environ.get("FOOTBALL_PREDICTOR_API_URL") or _secrets_get("API_BASE_URL", "http://localhost:8001")
API_KEY = os.environ.get("FOOTBALL_PREDICTOR_API_KEY") or _secrets_get("API_KEY", "demo-pro-key")

OUTCOME_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}

# Real-sport codes understood by the Sportscore endpoint.
SPORTS: dict[str, str] = {
    "Football": "football",
    "Basketball": "basketball",
    "Tennis": "tennis",
    "Ice Hockey": "ice_hockey",
    "Volleyball": "volleyball",
    "Baseball": "baseball",
    "Cricket": "cricket",
    "Table Tennis": "table_tennis",
}

VIRTUAL_SPORTS = {f"Virtual {s}": f"virtual_{s.lower()}" for s in SPORTS.keys()}
# Virtual-sport codes accepted by VirtualSportsEngine; key=display label, value=sport_name
# passed to /virtual/{sport_name}/upcoming (the engine prepends "virtual_" internally).
VIRTUAL_SPORTS: dict[str, str] = {
    "Football": "football",
    "Basketball": "basketball",
    "Tennis": "tennis",
    "Baseball": "baseball",
    "Ice Hockey": "hockey",
}

SPORT_LABELS = {code: label for label, code in SPORTS.items()}
SPORT_LABELS.update({code: label for label, code in VIRTUAL_SPORTS.items()})


def api_get(path: str, params: dict | None = None) -> object:
    resp = requests.get(
        f"{API_BASE_URL}{path}", headers={"X-API-Key": API_KEY}, params=params, timeout=10
    )
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, body: dict) -> object:
    resp = requests.post(
        f"{API_BASE_URL}{path}", headers={"X-API-Key": API_KEY}, json=body, timeout=10
    )
    resp.raise_for_status()
    return resp.json()


def _http_status(exc: requests.HTTPError) -> int | None:
    return exc.response.status_code if exc.response is not None else None


def probability_bar(p_home: float, p_draw: float, p_away: float, labels=("Home", "Draw", "Away")) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=[p_home, p_draw, p_away],
            y=list(labels),
            orientation="h",
            marker_color=["#1f77b4", "#999999", "#d62728"],
            text=[f"{v:.0%}" for v in (p_home, p_draw, p_away)],
            textposition="auto",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        height=160,
        margin=dict(l=0, r=0, t=10, b=10),
    )
    return fig


def _booking_code_widget(pred: dict, sport: str) -> None:
    """Render a 'Get Booking Code' expander inside a prediction card."""
    external_id = pred.get("external_id", "")
    with st.expander("Get SportyBet Booking Code"):
        # --- Booking Code Button ---
        if st.button(f"Get Booking Code", key=f"code_{external_id}"):
            payload = {
                "match_id": external_id,
                "prediction": pred.get("predicted_outcome", "H"),  # "H", "D", or "A"
                "odds": 2.10,
                "sport": sport.lower()
            }
            headers = {"X-API-Key": "demo-pro-key"}  # later replace with real key
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/api/v1/bookmaker/sportybet/convert",
                    json=payload,
                    headers=headers,
                    timeout=10
                )
                if resp.status_code == 200:
                    code = resp.json().get("booking_code")
                    st.success(f"Booking Code: **{code}**")
                else:
                    st.error(f"Failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")


def _render_prediction_cards(predictions: list, sport: str, show_booking: bool = True) -> None:
    """Shared card renderer for both real and virtual prediction lists."""
    for pred in predictions:
        kickoff = pd.to_datetime(pred["kickoff_utc"])
        participants = pred.get("participants") or {}
        participant_names = list(participants.values())
        p1 = participant_names[0] if len(participant_names) > 0 and participant_names[0] else "Player 1"
        p2 = participant_names[1] if len(participant_names) > 1 and participant_names[1] else "Player 2"

        with st.container(border=True):
            st.subheader(f"{p1} vs {p2}")
            st.caption(kickoff.strftime("%a %d %b %Y, %H:%M UTC"))

            col_chart, col_meta = st.columns([3, 1])
            with col_chart:
                if pred.get("home_win") is not None:
                    st.plotly_chart(
                        probability_bar(pred["home_win"], pred["draw"], pred["away_win"], labels=(p1, "Draw", p2)),
                        use_container_width=True,
                    )
                else:
                    st.write("Probabilities not included in your tier.")
            with col_meta:
                outcome = pred.get("predicted_outcome") or "?"
                st.metric("Predicted", OUTCOME_LABELS.get(outcome, outcome))
                if pred.get("confidence") is not None:
                    st.metric("Confidence", f"{pred['confidence']:.0%}")
                if pred.get("source"):
                    st.caption(f"Source: {pred['source'].replace('_', ' ')}")

            consensus = pred.get("consensus")
            if consensus and consensus["total"]:
                st.progress(consensus["agreeing"] / consensus["total"])
                st.caption(f"LLM committee consensus: {consensus['agreeing']}/{consensus['total']} agents agree")

            if pred.get("agent_opinions"):
                with st.expander("Individual agent predictions (elite)"):
                    for opinion in pred["agent_opinions"]:
                        st.markdown(f"**{opinion['agent']}**")
                        st.plotly_chart(
                            probability_bar(
                                opinion["home_win"], opinion["draw"], opinion["away_win"], labels=(p1, "Draw", p2)
                            ),
                            use_container_width=True,
                        )
                        st.caption(opinion["reasoning"])

            if show_booking:
                _booking_code_widget(pred, sport)


def _virtual_predictions_section(sport: str, sport_label: str) -> None:
    num = st.slider("Fixtures to generate", min_value=1, max_value=50, value=10)
    try:
        predictions = api_get(f"/api/v1/virtual/{sport}/upcoming", params={"num": num})
    except requests.HTTPError as exc:
        status_code = _http_status(exc)
        if status_code == 404:
            st.warning(f"No virtual {sport_label} league available.")
        else:
            st.error(f"Could not load virtual predictions: {exc}")
        return
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
        return

    if not predictions:
        st.info("No virtual fixtures generated.")
        return

    _render_prediction_cards(predictions, sport, show_booking=True)


def upcoming_predictions_page(sport: str, sport_label: str, mode: str = "Real Sports", bookmaker: str = "SportyBet") -> None:
    st.header(f"{mode} Predictions - {sport_label}")

    if mode == "Virtual Sports":
        sport_key = f"virtual_{sport.lower()}"
        url = f"{API_BASE_URL}/api/v1/virtual/{sport_key}/upcoming?num=10"
    else:
        sport_lower = sport.lower()
        url = f"{API_BASE_URL}/api/v1/sportscore/{sport_lower}/upcoming?days_ahead=7"

    try:
        resp = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        predictions = data.get("predictions", []) if isinstance(data, dict) else data
        if not predictions:
            st.info("No upcoming fixtures found for this sport.")
        else:
            __render_predictor_cards(predictions, sport, show_booking=True, bookmaker=bookmaker)
    except requests.HTTPError as exc:
        status_code = exc.response.status_code
        if status_code == 403:
            st.warning(f"The {sport_label} market requires a pro or elite tier API key.")
        elif status_code == 503:
            st.error("The sports data service is currently unavailable. Please try again later.")
        else:
            st.error(f"Error fetching predictions: {exc}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


def _football_accuracy_detail() -> None:
    """Detail view backed by the trained XGBoost model (football only)."""
    try:
        summary = api_get("/api/v1/accuracy/summary")
    except requests.HTTPError as exc:
        if _http_status(exc) == 404:
            st.info("No active football model yet.")
            return
        st.error(f"Could not load football accuracy summary: {exc}")
        return
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
        return

    st.subheader("Football model detail")

    if summary["n_predictions"] == 0:
        st.info(
            "No scored predictions yet. Generate predictions, wait for matches to "
            "finish, then run scripts/update_accuracy.py."
        )
        return

    cols = st.columns(4)
    cols[0].metric("Active model", summary["model_name"])
    cols[1].metric("Accuracy", f"{summary['accuracy']:.1%}")
    if summary.get("log_loss") is not None:
        cols[2].metric("Log loss", f"{summary['log_loss']:.3f}")
    if summary.get("brier_score") is not None:
        cols[3].metric("Brier score", f"{summary['brier_score']:.3f}")

    if summary.get("by_outcome"):
        st.markdown("**Accuracy by predicted outcome**")
        by_outcome_df = pd.DataFrame(
            [
                {"outcome": OUTCOME_LABELS.get(k, k), "accuracy": v["accuracy"], "n": v["n_predicted"]}
                for k, v in summary["by_outcome"].items()
            ]
        )
        st.bar_chart(by_outcome_df.set_index("outcome")["accuracy"])

    try:
        history = api_get("/api/v1/accuracy/history")
    except requests.RequestException:
        history = []

    if history:
        hist_df = pd.DataFrame(history)
        hist_df["kickoff_utc"] = pd.to_datetime(hist_df["kickoff_utc"])
        hist_df = hist_df.sort_values("kickoff_utc").reset_index(drop=True)
        hist_df["rolling_accuracy"] = hist_df["correct"].expanding().mean()

        st.markdown("**Rolling accuracy over time**")
        st.line_chart(hist_df.set_index("kickoff_utc")["rolling_accuracy"])

        st.markdown("**Confusion matrix (rows = predicted, columns = actual)**")
        confusion = pd.crosstab(hist_df["predicted_outcome"], hist_df["actual_outcome"])
        st.dataframe(confusion, use_container_width=True)

        st.markdown("**Most recent results**")
        st.dataframe(
            hist_df[["kickoff_utc", "home_team", "away_team", "predicted_outcome", "actual_outcome", "correct"]]
            .sort_values("kickoff_utc", ascending=False)
            .head(20),
            use_container_width=True,
            hide_index=True,
        )

    try:
        calibration = api_get("/api/v1/accuracy/calibration")
    except requests.RequestException:
        calibration = None

    if calibration and calibration.get("plot"):
        st.markdown("**Held-out test calibration (XGBoost model)**")
        st.caption(f"Held-out test set: {calibration['test_size']} matches")
        st.plotly_chart(go.Figure(calibration["plot"]), use_container_width=True)


def _sport_calibration_section(sport: str, sport_label: str) -> None:
    """Live calibration curve from tracked SportPrediction outcomes (elite tier only)."""
    try:
        calibration = api_get(f"/api/v1/sports/{sport}/calibration")
    except requests.HTTPError as exc:
        if _http_status(exc) == 403:
            return  # Elite-only: say nothing to lower tiers.
        st.error(f"Could not load {sport_label} calibration: {exc}")
        return
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
        return

    st.subheader(f"Live calibration — {sport_label} (elite)")
    if calibration.get("plot"):
        st.caption(f"Resolved predictions: {calibration['n_resolved']}")
        st.plotly_chart(go.Figure(calibration["plot"]), use_container_width=True)
    else:
        st.info("Not enough resolved predictions yet for a calibration curve.")


def accuracy_tracking_page(sport: str, sport_label: str) -> None:
    st.header(f"Accuracy Tracking — {sport_label}")

    try:
        leaderboard = api_get("/api/v1/sports/leaderboard")
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
        return

    entries = {e["sport"]: e for e in leaderboard["sports"]}
    entry = entries.get(sport)

    if entry and entry["n_resolved"] > 0:
        cols = st.columns(2)
        cols[0].metric("Resolved predictions", entry["n_resolved"])
        cols[1].metric("Overall accuracy", f"{entry['overall_accuracy']:.1%}")

        st.subheader("Accuracy by tier")
        tier_df = pd.DataFrame(
            [
                {
                    "tier": tier["tier"],
                    "min_confidence": tier["min_confidence"],
                    "n_samples": tier["n_samples"],
                    "accuracy": tier["accuracy"],
                    "coverage_pct": tier["coverage_pct"],
                }
                for tier in entry["tiers"].values()
            ]
        )
        st.dataframe(tier_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            f"No resolved {sport_label} predictions yet. Predictions made on the "
            "'Upcoming Predictions' page are tracked automatically and will appear "
            "here once those fixtures are played and scored."
        )

    st.subheader("Which sports is this system best at?")
    overview_df = pd.DataFrame(
        [
            {"sport": SPORT_LABELS.get(e["sport"], e["sport"]), "accuracy": e["overall_accuracy"]}
            for e in leaderboard["sports"]
        ]
    )
    st.bar_chart(overview_df.set_index("sport")["accuracy"])

    if sport == "football":
        _football_accuracy_detail()

    _sport_calibration_section(sport, sport_label)


def main() -> None:
    st.set_page_config(page_title="Sport Predictor", layout="wide")
    st.sidebar.title("Sport Predictor")
    st.sidebar.caption(f"API: {API_BASE_URL}")

    mode = st.sidebar.radio("Mode", ["Real Sports", "Virtual Sports"])

    if mode == "Virtual Sports":
        sport_label = st.sidebar.selectbox("Sport", list(VIRTUAL_SPORTS.keys()))
        sport = VIRTUAL_SPORTS[sport_label]
    else:
        sport_label = st.sidebar.selectbox("Sport", list(SPORTS.keys()))
        sport = SPORTS[sport_label]

    # --- ADD BOOKMAKER SELECTOR ---
    bookmaker = st.sidebar.selectbox("Bookmaker", ["SportyBet", "Bet9ja", "1xBet"])

    # --- ADD "Live In-Play" to page options ---
    page = st.sidebar.radio("Page", ["Upcoming Predictions", "Accuracy Tracking", "Live In-Play"])

    if page == "Upcoming Predictions":
        upcoming_predictions_page(sport, sport_label, mode, bookmaker)
    elif page == "Accuracy Tracking":
        accuracy_tracking_page(sport, sport_label)
    elif page == "Live In-Play":
        # --- MOVED LIVE IN-PLAY BLOCK INSIDE main() ---
        st.subheader("🏟️Live Match Tracker")
        fixture_id = "live_123"

        if "live_state" not in st.session_state:
            st.session_state.live_state = {
                "clock": 0,
                "home_score": 0,
                "away_score": 0,
                "home_shots": 0,
                "away_shots": 0,
                "momentum": 0,
                "prediction": None,
                "events": []
            }
        state = st.session_state.live_state

        col1, col2, col3 = st.columns(3)
        col1.metric("⏱ Minute", state["clock"])
        col2.metric(" Home", state["home_score"])
        col3.metric(" Away", state["away_score"])

        if state["prediction"]:
            st.info(f" Live Prediction: {state['prediction']}")

        if st.button(" Advance 1 minute"):
            import random
            state["clock"] += 1

            event = None
            if random.random() < 0.2:
                if random.random() < 0.5:
                    state["home_score"] += 1
                    event = f" Goal! Home scores! ({state['home_score']}-{state['away_score']})"
                    st.balloons()
                else:
                    state["away_score"] += 1
                    event = f" Goal! Away scores! ({state['home_score']}-{state['away_score']})"
                    st.balloons()
            else:
                if random.random() < 0.3:
                    event = " Possession change"
                elif random.random() < 0.5:
                    event = " Shot on target"
                else:
                    event = " Corner kick"

            if event:
                state["events"].append(f"Min {state['clock']}: {event}")
                st.write(f"**Event:** {event}")

            # Simple prediction update
            if state["home_score"] > state["away_score"]:
                pred = f"Home win {state['home_score']}-{state['away_score']} (confidence {random.randint(65,95)}%)"
            elif state["home_score"] < state["away_score"]:
                pred = f"Away win {state['home_score']}-{state['away_score']} (confidence {random.randint(65,95)}%)"
            else:
                pred = f"Draw {state['home_score']}-{state['away_score']} (confidence {random.randint(55,80)}%)"
            state["prediction"] = pred

            st.rerun()

        if state["events"]:
            st.subheader(" Event Log")
            for e in state["events"][-5:]:
                st.write(e)

        st.caption("Simulated live match. Click the button to advance time.")
