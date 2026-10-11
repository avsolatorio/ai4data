#!/usr/bin/env bash
# Multilingual base model, data side: plan per-language budgets (FineWeb-2 languages with >= 10 MB, plus English
# Wikipedia), fetch them, then train tokenizers of several sizes and compare bytes per token by language.
# Every step skips when its output exists, so a restart resumes.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="$HOME/venvs/relexp/bin/python"
R=data/corpus_multi
MC="$PY -m relevance_explainer.tiny.multicorpus"
echo "multibase start $(date -u +%H:%M)"
[ -f "$R/plan.json" ] || $MC plan --sizes /data/wb536061/ai4data-relevance-explainer/multibase/fw2_size_hub.json --root "$R" --chars 18e9 || exit 1
for attempt in 1 2 3; do  # failed languages are retried; after three attempts the rest goes ahead without them
  grep -q FETCH-DONE "$R/fetch.log" 2>/dev/null && break
  $MC fetch --root "$R" --procs 24 2>&1 | grep -v Warning | tee "$R/fetch.log"
done
echo "fetched $(date -u +%H:%M)"
[ -f "$R/tokenizers/bytes_per_token.json" ] || $MC tokenizers --root "$R" --vocab 8000 12000 16000 24000 32000 \
  --compare data/corpus/tokenizer.json --task-data data/v2_g_itemfirst_nosim --sample-chars 2e9 --max-lang-chars 40e6 \
  2>&1 | tee "$R/tokenizers.log" || exit 1
echo "MULTIBASE-DATA-DONE $(date -u +%H:%M)"
