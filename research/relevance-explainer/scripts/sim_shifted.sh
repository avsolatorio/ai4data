#!/usr/bin/env bash
# Score a run on the simulated retriever-scale variants after scripts/sim_variants.sh. Usage: scripts/sim_shifted.sh [run]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-runs/v2b-cont-itemfirst-dpo}
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
PY="$HOME/venvs/relexp/bin/python"
until grep -q SIM-VARIANTS-DONE /tmp/claude-1002/-home-jupyter-wb536061-WBG-ai4data/f9459d82-c4fb-49c4-a88f-41cc1ffd548f/scratchpad/sim_variants.log; do sleep 20; done
$PY scripts/sim_shifted.py
for v in shift_up compress generic_like; do
  $PY -m relevance_explainer.tiny.evaluate --run "$RUN" --data "data/sim_variants/$v" --precisions int8 --tag "sim_$v" --script-guard > /dev/null
  echo "$v: $($PY -m relevance_explainer.synth.score_real --data "data/sim_variants/$v" --run "$RUN" --tag "sim_$v" | head -1)"
done
echo SIM-SHIFTED-DONE
