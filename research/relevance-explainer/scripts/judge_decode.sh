#!/usr/bin/env bash
# Judge runs' realistic-test answers from the decode test (greedy, no repeat ban) with the teacher: the explanation
# (tiny/judge.py, 800 examples, the same for every run) and, for tagged runs, the tl;dr (scripts/judge_tldr.py).
#   scripts/judge_decode.sh [--n 800] runs/fmt-s-A runs/fmt-s-B runs/fmt-s-C
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' LLM_REASONING=
export LLM_BASE_URL=http://localhost:8000/v1,http://localhost:8001/v1
PY="$HOME/venvs/relexp/bin/python"
N=800
if [ "${1:-}" = "--n" ]; then N=$2; shift 2; fi
for R in "$@"; do
  $PY scripts/judge_prep_decode.py "$R" data/eval_real/itemfirst_nosim_v4 --tag realnb || continue
  [ -f "$R/judge_realnb_int8.json" ] || $PY -m relevance_explainer.tiny.judge --run "$R" --prec int8 --tag realnb --n "$N" --workers 48 | grep -v "^  "
  if grep -q '<|tldr|>' "$R/realnb_generations_int8.jsonl"; then
    [ -f "$R/judge_tldr_realnb.json" ] || $PY scripts/judge_tldr.py "$R" --tag realnb --n "$N" --workers 48
  fi
done
echo "JUDGE-DECODE-DONE $(date -u +%H:%M)"
