#!/usr/bin/env bash
# Create missing translation sets tK for even K whose chunks cK-1 and cK exist (20% of the two chunks).
# Usage: scripts/translate_backfill.sh [max_k=200]
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
for k in $(seq 2 2 "${1:-200}"); do
  a="data/real/c$((k - 1))/examples.jsonl"; b="data/real/c$k/examples.jsonl"
  if [ -f "$a" ] && [ -f "$b" ] && [ ! -f "data/real/t$k/examples.jsonl" ]; then
    n=$(( ($(wc -l < "$a") + $(wc -l < "$b")) / 5 ))
    "$RELEXP_PY" -m relevance_explainer.synth.translate --src "data/real/c$((k - 1))" "data/real/c$k" --out "data/real/t$k" --n "$n" --seed "$k" --workers 24 \
      && echo "t$k: $(wc -l < "data/real/t$k/examples.jsonl")"
  fi
done
echo BACKFILL-DONE
