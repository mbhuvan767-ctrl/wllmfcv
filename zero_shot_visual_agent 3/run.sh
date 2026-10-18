#!/usr/bin/env bash
# One-click launcher (macOS / Linux): creates a virtual env, installs dependencies, starts the app.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -d .venv ] || { echo "Creating virtual environment..."; $PY -m venv .venv; }
source .venv/bin/activate
if [ ! -f .venv/.installed ]; then
  echo "Installing dependencies (first time only, a few minutes)..."
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  touch .venv/.installed
fi
echo "Starting app at http://localhost:8501 ..."
streamlit run app.py
