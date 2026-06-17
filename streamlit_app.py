"""Root-level entry-point for Streamlit Cloud / local `streamlit run streamlit_app.py`.

Delegates entirely to the dashboard module so there is one source of truth.
Run locally:
    streamlit run streamlit_app.py
Or point Streamlit Cloud's main-file setting at this file.
"""
from football_predictor.dashboard.app import main

main()
