#!/usr/bin/env bash
# Launch the robo-classifier review UI.
# Opens http://localhost:8765/ in your default browser.

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

PORT="${PORT:-8765}"

# Open browser after a moment
(sleep 1.5 && open "http://localhost:${PORT}/" 2>/dev/null || true) &

exec python ui/app.py --port "$PORT" "$@"
