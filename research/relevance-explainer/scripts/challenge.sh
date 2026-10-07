#!/usr/bin/env bash
# Challenge suite: write, label, build, then score the current realistic models per phenomenon. Usage: scripts/challenge.sh [per=150]
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' LLM_REASONING=
PY="$HOME/venvs/relexp/bin/python"
[ -f data/challenge/searches.jsonl ] || $PY -m relevance_explainer.synth.challenge write --out data/challenge --per "${1:-150}" || exit 1
[ -f data/challenge/labeled.jsonl ] || $PY -m relevance_explainer.synth.challenge label --out data/challenge || exit 1
$PY -m relevance_explainer.synth.challenge build --out data/challenge --eval data/eval_challenge || exit 1
for pair in "runs/v2c-cont-itemfirst-nosim-dpo itemfirst_nosim" "runs/v2c-cont-itemfirst-dpo itemfirst"; do
  set -- $pair
  $PY -m relevance_explainer.tiny.evaluate --run "$1" --data "data/eval_challenge/$2" --precisions int8 --tag challenge --script-guard > /dev/null 2>&1
  $PY -m relevance_explainer.synth.score_real --data "data/eval_challenge/$2" --run "$1" --tag challenge
done
echo CHALLENGE-DONE
