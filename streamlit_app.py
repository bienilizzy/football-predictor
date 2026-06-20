import sys
import os

# Add the src/ folder to Python's path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import traceback

st.set_page_config(page_title="Sport Predictor", layout="wide")

try:
    from football_predictor.dashboard.app import main
    main()
except Exception as e:
    st.error("Error loading dashboard:")
    st.code(traceback.format_exc())
