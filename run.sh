#!/usr/bin/env bash
# Starts Shhhhhhh AI and opens it in your browser (Chrome recommended).
set -euo pipefail
cd "$(dirname "$0")"
PORT="${MAUN_PORT:-8765}"
[ -x .venv/bin/python ] || { echo "Run ./setup.sh first"; exit 1; }
( sleep 2; open "http://127.0.0.1:$PORT" 2>/dev/null || true ) &
exec .venv/bin/python -m server.app
