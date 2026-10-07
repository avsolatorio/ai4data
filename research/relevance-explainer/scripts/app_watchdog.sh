#!/usr/bin/env bash
# Restart the web app when its health check fails. Run from cron, e.g. every 5 minutes:
#   */5 * * * * /path/to/research/relevance-explainer/scripts/app_watchdog.sh
cd "$(dirname "$0")/.."
LOG=${APP_LOG:-/data/wb536061/ai4data-relevance-explainer/app_3410.log}
if ! curl -s -o /dev/null --max-time 10 -w "%{http_code}" "http://localhost:${PORT:-3410}/healthz" | grep -q 200; then
  pgrep -u "$(id -u)" -f "uvicorn relevance_explainer.webapp.server" > /dev/null && exit 0  # starting up
  echo "$(date -u +%FT%TZ) health check failed; restarting" >> "$LOG"
  setsid nohup scripts/serve_app.sh >> "$LOG" 2>&1 < /dev/null &
fi
