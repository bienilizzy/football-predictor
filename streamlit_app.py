import streamlit as st
import sys
import traceback

st.set_page_config(page_title="Sport Predictor", layout="wide")

try:
    st.write("Loading...")
    from football_predictor.dashboard.app import main
    st.write("Imported successfully!")
    main()
except Exception as e:
    st.error("Error loading dashboard:")
    st.code(traceback.format_exc())
