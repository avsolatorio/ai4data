#!/usr/bin/env bash
# No-score model: on-policy DPO, realistic-test scoring and judging without the similarity line, then the
# retrieval-score sensitivity variants. Usage: scripts/nosim_followup.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="$HOME/venvs/relexp/bin/python"
[ -f data/eval_real/itemfirst_nosim/test.jsonl ] || $PY scripts/strip_similarity.py data/eval_real/itemfirst data/eval_real/itemfirst_nosim || exit 1
scripts/dpo_run.sh runs/v2c-cont-itemfirst-nosim data/v2_c_itemfirst_nosim itemfirst_nosim || exit 1
scripts/sim_variants.sh runs/v2c-cont-itemfirst-nosim-dpo
SIMLOG=/dev/null scripts/sim_shifted.sh runs/v2c-cont-itemfirst-nosim-dpo
echo NOSIM-FOLLOWUP-DONE
