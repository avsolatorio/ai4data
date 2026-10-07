#!/usr/bin/env bash
# keepalive.sh for round E, logging to the data disk. Start with: tmux new -d -s relexp scripts/keepalive_round_e.sh
cd "$(dirname "$0")/.."
scripts/keepalive.sh scripts/round_e.sh ROUND-E-DONE 2>&1 | tee -a /data/wb536061/ai4data-relevance-explainer/keepalive.log
