#!/usr/bin/env bash
# Paraphrase test set (anchored searches only, every fully-answering one paraphrased) and the current models' scores.
# Usage: scripts/para_test.sh [n_anchored=2500]
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 PARAPHRASE_SHARE=1
export LLM_BASE_URL=http://localhost:8000/v1,http://localhost:8001/v1 QUERY_MODEL=qwen3.8-27b NARRATE_MODEL=qwen3.8-27b LLM_REASONING=
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
PY="$RELEXP_PY"
OUT=data/para_test
mkdir -p "$OUT"
[ -f "$OUT/queries.jsonl" ] || : > "$OUT/queries.jsonl"
[ -f "$OUT/anchored.jsonl" ] || $PY -m relevance_explainer.synth.realistic anchored --out "$OUT" --n "${1:-2500}" --seed 9001 --themes data/themes.json --retrieve-workers 20 || exit 1
[ -f "$OUT/pairs.jsonl" ] || $PY -m relevance_explainer.synth.realistic retrieve --out "$OUT" --seed 9001 --retrieve-workers 20 || exit 1
[ -f "$OUT/rendered.jsonl" ] || $PY -m relevance_explainer.synth.realistic render --out "$OUT" || exit 1
[ -f "$OUT/examples.jsonl" ] || $PY -m relevance_explainer.synth.realistic narrate --out "$OUT" --workers 120 || exit 1
[ -f data/eval_para/itemfirst/test.jsonl ] || $PY -m relevance_explainer.synth.para_eval --src "$OUT" --out data/eval_para || exit 1
for pair in "runs/v2c-cont-itemfirst-nosim-dpo itemfirst_nosim" "runs/v2c-cont-itemfirst-dpo itemfirst"; do
  set -- $pair
  $PY -m relevance_explainer.tiny.evaluate --run "$1" --data "data/eval_para/$2" --precisions int8 --tag para --script-guard > /dev/null 2>&1
  $PY -m relevance_explainer.synth.score_real --data "data/eval_para/$2" --run "$1" --tag para
done
echo PARA-TEST-DONE
