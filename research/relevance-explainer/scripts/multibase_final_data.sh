#!/usr/bin/env bash
# Multilingual base, data for the final run: about 80B training tokens (176B characters at the 32k tokenizer's
# 2.2 characters per token), English at a fixed 10% (Wikipedia and FineWeb-Edu), the other languages by size**0.3
# with at most 4 passes over a language's text.
#   1. plan with each language asking for its whole budget (--fetch-full)
#   2. fetch (three attempts; failures are retried)
#   3. reweight the budgets from the characters actually fetched
#   4. a 32k tokenizer on a large balanced sample, compared with the 32k tokenizer of the shape runs
#   5. tokenize
# Every step skips when its output exists, so a restart resumes.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="$HOME/venvs/relexp/bin/python"
R=data/corpus_multi_final
MC="$PY -m relevance_explainer.tiny.multicorpus"
echo "final data start $(date -u +%H:%M)"
[ -f "$R/plan_target.json" ] || [ -f "$R/plan.json" ] || $MC plan --sizes /data/wb536061/ai4data-relevance-explainer/multibase/fw2_size_hub.json \
  --root "$R" --chars 176e9 --english-share 0.10 --fetch-full || exit 1
for attempt in 1 2 3; do
  grep -q FETCH-DONE "$R/fetch.log" 2>/dev/null && break
  $MC fetch --root "$R" --procs 32 2>&1 | grep --line-buffered -v Warning | tee "$R/fetch.log"
done
echo "fetched $(date -u +%H:%M)"
[ -f "$R/plan_target.json" ] || $MC reweight --root "$R" --max-epochs 4 2>&1 | tee "$R/reweight.log" || exit 1
[ -f "$R/tokenizers/bytes_per_token.json" ] || $MC tokenizers --root "$R" --vocab 32000 \
  --compare data/corpus_multi_v32k/tokenizer.json --task-data data/v2_g_itemfirst_nosim --sample-chars 25e9 \
  --max-lang-chars 400e6 2>&1 | tee "$R/tokenizers.log" || exit 1
[ -f "$R/tokenizer.json" ] || cp "$R/tokenizers/bpe32000.json" "$R/tokenizer.json"
[ -f "$R/tok/.done" ] || { $PY -m relevance_explainer.tiny.corpus tokenize --root "$R" --procs 128 | tail -1 && touch "$R/tok/.done"; } || exit 1
echo "FINAL-DATA-DONE $(date -u +%H:%M)"
