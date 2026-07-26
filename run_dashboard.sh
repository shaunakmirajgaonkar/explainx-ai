#!/usr/bin/env bash
# Starts the ExplainX AI Streamlit dashboard on http://127.0.0.1:8501
# Make sure run_backend.sh is already running in another terminal.
set -e
cd "$(dirname "$0")"
streamlit run dashboard/app.py --server.port 8501
