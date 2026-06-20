#!/bin/bash
export PYTHONPATH=/opt/render/project/src
export FOOTBALL_PREDICTOR_API_URL=${FOOTBALL_PREDICTOR_API_URL:-"https://football-predictor-api.onrender.com"}
streamlit run src/football_predictor/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
