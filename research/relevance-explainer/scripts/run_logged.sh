#!/usr/bin/env bash
# Run a script with its output appended to <log_dir>/<script name>.log, for use as a tmux window command:
#   tmux new-window -t relexp -n roundf scripts/run_logged.sh scripts/round_f.sh
cd "$(dirname "$0")/.."
LOGD=${LOGD:-/data/wb536061/ai4data-relevance-explainer}
"$@" >> "$LOGD/$(basename "$1" .sh).log" 2>&1
