#!/usr/bin/env bash
# The DPO round for full-data C with the judge of full-data B in the same teacher window:
# sample (teachers off), teachers up, judge runs/final-B, then scripts/dpo_c.sh (scores, takes the teachers down for
# training and the test sets, brings them up for its own judge). Steps skip when their outputs exist.
#   tmux new-window -d -t relexp -n dpo 'scripts/dpo_round_c.sh > runs/final-C-dpo.pipeline.log 2>&1'
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src
PY="$RELEXP_PY"
D=data/final/C; S=onpolicy_c_final-C_samples.jsonl
[ -f "$D/$S" ] || CUDA_VISIBLE_DEVICES=0 $PY -m relevance_explainer.distill.onpolicy_c sample --run runs/final-C --data "$D" \
  --n 15000 --k 8 --out "$S" || exit 1
if [ ! -f runs/final-B/judge_realnb_int8.json ]; then
  scripts/teachers.sh up || exit 1
  scripts/judge_decode.sh runs/final-B
  $PY scripts/judge_paired.py runs/final-B runs/final-C
fi
scripts/dpo_c.sh runs/final-C "$D" 15000 8 0
echo "DPO-ROUND-C-DONE $(date -u +%H:%M)"
