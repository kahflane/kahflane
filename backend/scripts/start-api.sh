#!/bin/bash
# Start the Kahflane API server.
# Run from backend/ directory.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BACKEND_DIR"

# Kill any existing uvicorn instances serving this app
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 1

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
RELOAD="${RELOAD:-true}"

if [ "$RELOAD" = "true" ]; then
    exec ./venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
else
    exec ./venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
fi
