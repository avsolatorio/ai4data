#!/usr/bin/env bash
# Run scripts/keepalive.sh with its log appended to the shared log directory. Usage: tmux new -d -s relexp scripts/run_keepalive.sh
cd "$(dirname "$0")/.."
scripts/keepalive.sh "$@" 2>&1 | tee -a /data/wb536061/ai4data-relevance-explainer/keepalive.log
