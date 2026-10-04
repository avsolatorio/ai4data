#!/usr/bin/env bash
# Judge a run's realistic-test narratives with the teacher and Gemma 4 (800 examples). Usage: scripts/judge_real.sh <run>
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${1:?run}
export PYTHONPATH=src LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
PY="$HOME/venvs/relexp/bin/python"
LLM_BASE_URL=http://localhost:8000/v1,http://localhost:8001/v1 $PY -m relevance_explainer.tiny.judge --run "$RUN" --prec int8 --tag realtest --n 800 --workers 48
LLM_BASE_URL=http://localhost:8002/v1 $PY -m relevance_explainer.tiny.judge --run "$RUN" --prec int8 --tag realtest --n 800 --workers 16 --model gemma-4-31b --out-suffix _gemma4
echo "JUDGE-REAL-DONE $RUN"
