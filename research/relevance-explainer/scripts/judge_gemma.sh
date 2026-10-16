#!/usr/bin/env bash
# Held-out judge: Gemma 4 judges the realistic-test answers that the teacher judged (tiny/judge.py, same 800 examples,
# <run>/realnb_generations_int8.jsonl from scripts/judge_prep_decode.py), then paired comparisons. Gemma and the GPU 0
# teacher run only for this script (scripts/teachers.sh gemma-up / gemma-down).
#   scripts/judge_gemma.sh runs/final-B runs/final-C runs/final-C-dpo
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src LLM_BASE_URL=http://localhost:8002/v1
PY="$RELEXP_PY"
todo=0
for R in "$@"; do [ -f "$R/judge_realnb_int8_gemma4.json" ] || todo=1; done
if [ $todo = 1 ]; then
  scripts/teachers.sh gemma-up || exit 1
  for R in "$@"; do
    [ -f "$R/judge_realnb_int8_gemma4.json" ] || $PY -m relevance_explainer.tiny.judge --run "$R" --prec int8 --tag realnb \
      --n 800 --workers 32 --model gemma-4-31b --out-suffix _gemma4 | grep -v "^  "
  done
  scripts/teachers.sh gemma-down
fi
prev=""
for R in "$@"; do
  [ -n "$prev" ] && $PY scripts/judge_paired.py "$prev" "$R" --suffix _gemma4
  prev=$R
done
echo "JUDGE-GEMMA-DONE $(date -u +%H:%M)"
