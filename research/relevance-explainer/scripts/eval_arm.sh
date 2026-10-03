#!/usr/bin/env bash
# Evaluate one run on the standard test set (data/f2, INT8) and judge 800 generations with the teacher and with
# Gemma 4. Usage: scripts/eval_arm.sh <run> [skip-generate]
# Environment: GPU (default 1), DATA (test-set directory, default data/f2), TEACHER_URLS, GEMMA_URL (default http://localhost:8002/v1).
set -euo pipefail
cd "$(dirname "$0")/.."
run=$1
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTHONPATH=src
py="$HOME/venvs/relexp/bin/python"
if [ -z "${2:-}" ]; then
  $py -m relevance_explainer.tiny.evaluate --run "$run" --data "${DATA:-data/f2}" --precisions int8 --tag f2test
fi
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
LLM_BASE_URL=${TEACHER_URLS:-http://localhost:8000/v1,http://localhost:8001/v1} \
  $py -m relevance_explainer.tiny.judge --run "$run" --prec int8 --tag f2test --n 800 --workers 64
LLM_BASE_URL=${GEMMA_URL:-http://localhost:8002/v1} \
  $py -m relevance_explainer.tiny.judge --run "$run" --prec int8 --tag f2test --n 800 --workers 64 \
  --model gemma-4-31b --out-suffix _gemma4
echo EVAL-ARM-DONE "$run"
