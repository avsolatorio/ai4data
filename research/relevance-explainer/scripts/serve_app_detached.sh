#!/usr/bin/env bash
# Start scripts/serve_app.sh detached from the calling shell (survives the session). Usage: scripts/serve_app_detached.sh <log>
cd "$(dirname "$0")/.."
setsid nohup scripts/serve_app.sh > "${1:-app_3410.log}" 2>&1 < /dev/null &
echo "started pid $!"
