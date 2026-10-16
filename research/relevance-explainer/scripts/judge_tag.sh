#!/usr/bin/env bash
# Judge a run's generations for a tag with the teacher and Gemma 4 (800 examples). Usage: scripts/judge_tag.sh <run> <tag>
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
RUN=${1:?run}; TAG=${2:?tag}
export PYTHONPATH=src LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
PY="$RELEXP_PY"
LLM_BASE_URL=http://localhost:8000/v1,http://localhost:8001/v1 $PY -m relevance_explainer.tiny.judge --run "$RUN" --prec int8 --tag "$TAG" --n 800 --workers 48
LLM_BASE_URL=http://localhost:8002/v1 $PY -m relevance_explainer.tiny.judge --run "$RUN" --prec int8 --tag "$TAG" --n 800 --workers 16 --model gemma-4-31b --out-suffix _gemma4
echo "JUDGE-DONE $RUN $TAG"
