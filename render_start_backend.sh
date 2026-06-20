#!/bin/bash
export PYTHONPATH="/opt/render/project/src:$PYTHONPATH"
uvicorn football_predictor.api.main:app --host 0.0.0.0 --port 8001
