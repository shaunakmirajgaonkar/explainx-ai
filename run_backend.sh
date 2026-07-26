#!/usr/bin/env bash
# Starts the ExplainX AI FastAPI backend locally on http://127.0.0.1:8000
set -e
cd "$(dirname "$0")/backend"
uvicorn main:app --reload --host 127.0.0.1 --port 8000
