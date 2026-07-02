#!/usr/bin/env bash
# Launch the Air3 ingest tool.
# Opens http://localhost:8766/ in your default browser.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

PORT="${PORT:-8767}"

(sleep 1.5 && open "http://localhost:${PORT}/" 2>/dev/null || true) &

exec python air3_ingest/app.py --port "$PORT" "$@"
