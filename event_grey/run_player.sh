#!/usr/bin/env bash
# Convenience launcher for the graphical event-grayscale player.
# Usage: ./run_player.sh [extra event-gray-player args]
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x "../.venv/bin/python" ] && [ ! -x ".venv/bin/python" ]; then
  echo "virtualenv not found; create it with: python3 -m venv .venv && .venv/bin/pip install -e event_grey" >&2
  exit 1
fi
PY=".venv/bin/python"
[ -x "$PY" ] || PY="../.venv/bin/python"
exec "$PY" -m event_grey.player "$@"
