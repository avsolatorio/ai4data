#!/usr/bin/env bash
# Serve one model of synth/models.py with vLLM, logging to $RELEXP_LOGS/judge_<name>.log. Meant for a tmux window:
#   tmux new-window -t relexp -n nemotron 'scripts/serve_judge.sh nemotron-3.5-lightning --gpus 1'
# Stop it with Ctrl-C in that window, or: pkill -u "$(id -u)" -f "served-model-name <name>"
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
NAME=${1:?model name, see: python -m relevance_explainer.synth.models list}
"$RELEXP_PY" -m relevance_explainer.synth.models serve "$@" 2>&1 | tee -a "$RELEXP_LOGS/judge_$NAME.log"
