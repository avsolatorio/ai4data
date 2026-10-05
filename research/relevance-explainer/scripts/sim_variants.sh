#!/usr/bin/env bash
# Build the retrieval-score variants of the realistic test set and score a run on each. Usage: scripts/sim_variants.sh [run]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-runs/v2b-cont-itemfirst-dpo}
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
PY="$HOME/venvs/relexp/bin/python"
[ -f data/sim_variants/stats.json ] || $PY -m relevance_explainer.synth.sim_variants --out data/sim_variants
for v in orig none const_median shuffled ours_recomputed e5small mlminilm minilm_en; do
  $PY -m relevance_explainer.tiny.evaluate --run "$RUN" --data "data/sim_variants/$v" --precisions int8 --tag "sim_$v" --script-guard > /dev/null
  $PY -m relevance_explainer.synth.score_real --data "data/sim_variants/$v" --run "$RUN" --tag "sim_$v" | head -1
done
echo SIM-VARIANTS-DONE
