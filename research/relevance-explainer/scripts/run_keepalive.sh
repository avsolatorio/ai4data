#!/usr/bin/env bash
# Run scripts/keepalive.sh with its log appended to the shared log directory. Usage: tmux new -d -s relexp scripts/run_keepalive.sh
cd "$(dirname "$0")/.."
. scripts/env.sh
scripts/keepalive.sh "$@" 2>&1 | tee -a "$RELEXP_LOGS/keepalive.log"
