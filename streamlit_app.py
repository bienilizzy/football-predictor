import streamlit as st
import sys
import traceback

st.set_page_config(page_title="Sport Predictor", layout="wide")

try:
    # Try to import the real dashboard
    from football_predictor.dashboard.app import main
    main()
except Exception as e:
    st.error("Something went wrong loading the full dashboard.")
    st.code(traceback.format_exc())
    st.info("The backend API might be down or a module is missing.")
