#!/usr/bin/env bash
# Launch the German TTS app: sets up the venv on first run, starts the server,
# and opens the browser once it's ready. Ctrl+C stops it.
#
# Engine is selected by TTS_ENGINE (default: xtts). Each engine has its own venv
# and requirements so their dependencies don't collide:
#   xtts       -> .venv-xtts   (default; steadier, ~real-time on CPU)
#   chatterbox -> .venv        (more expressive, but slower + occasional artifacts)
set -euo pipefail
cd "$(dirname "$0")"

ENGINE="${TTS_ENGINE:-xtts}"
if [ "$ENGINE" = "chatterbox" ]; then
  VENV=".venv";      REQ="requirements.txt"
else
  VENV=".venv-xtts"; REQ="requirements-xtts.txt"
fi
PY="$VENV/bin/python"
URL="http://localhost:5050"

# First-run setup: create the Python 3.11 venv and install the engine's deps.
if [ ! -x "$PY" ]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' is required for first-time setup (https://github.com/astral-sh/uv)." >&2
    exit 1
  fi
  echo "First run ($ENGINE): creating $VENV (Python 3.11) and installing $REQ …"
  uv venv --python 3.11 "$VENV"
  uv pip install --python "$VENV" -r "$REQ"
fi

# Open the browser once the server answers (runs in the background).
(
  for _ in $(seq 1 120); do
    if curl -s "$URL/api/status" >/dev/null 2>&1; then
      open "$URL" 2>/dev/null || true   # macOS; harmless elsewhere
      break
    fi
    sleep 1
  done
) &

echo "Starting German TTS server ($ENGINE) at $URL  (Ctrl+C to stop)…"
TTS_ENGINE="$ENGINE" TTS_DEVICE="${TTS_DEVICE:-cpu}" exec "$PY" server.py
