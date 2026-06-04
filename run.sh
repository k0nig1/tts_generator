#!/usr/bin/env bash
# Launch the German TTS app: sets up the venv on first run, starts the server,
# and opens the browser once it's ready. Ctrl+C stops it.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/bin/python"
URL="http://localhost:5050"

# First-run setup: create the Python 3.11 venv and install dependencies.
if [ ! -x "$PY" ]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' is required for first-time setup (https://github.com/astral-sh/uv)." >&2
    exit 1
  fi
  echo "First run: creating virtual environment (Python 3.11) and installing deps…"
  uv venv --python 3.11 "$VENV"
  uv pip install --python "$VENV" -r requirements.txt
fi

# Open the browser once the server answers (runs in the background).
(
  for _ in $(seq 1 90); do
    if curl -s "$URL/api/status" >/dev/null 2>&1; then
      open "$URL" 2>/dev/null || true   # macOS; harmless elsewhere
      break
    fi
    sleep 1
  done
) &

echo "Starting German TTS server at $URL  (Ctrl+C to stop)…"
exec "$PY" server.py
