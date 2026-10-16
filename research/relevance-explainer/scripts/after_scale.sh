#!/usr/bin/env bash
# Wait for the first 10 chunks to finish, translate chunks 1-10 (20%), then start the long run.
cd "$(dirname "$0")/.."
. scripts/env.sh
LOG=${1:?log file of realistic_scale.sh}
until grep -q REALISTIC-SCALE-DONE "$LOG"; do sleep 120; done
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
for k in 2 4 6 8 10; do
  if [ ! -f "data/real/t$k/examples.jsonl" ] && [ -f "data/real/c$k/examples.jsonl" ]; then
    n=$(( ($(wc -l < data/real/c$((k - 1))/examples.jsonl) + $(wc -l < data/real/c$k/examples.jsonl)) / 5 ))
    "$RELEXP_PY" -m relevance_explainer.synth.translate --src "data/real/c$((k - 1))" "data/real/c$k" --out "data/real/t$k" --n "$n" --seed "$k" --workers 24
  fi
done
exec scripts/realistic_long.sh 11 200
