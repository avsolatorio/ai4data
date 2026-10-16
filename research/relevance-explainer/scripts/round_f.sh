#!/usr/bin/env bash
# Round F: round E's data without narratives that state numbers absent from the input; one epoch from round E's
# supervised model; DPO whose pairs prefer explanations without invented numbers; evaluations with and without resampling.
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
PY="$RELEXP_PY"
D=data/v2_f_itemfirst_nosim; RUN=runs/v2f-nosim
echo "round F start $(date -u +%H:%M)"
[ -f "$D/train.jsonl" ] || $PY scripts/filter_numbers.py data/v2_e_itemfirst_nosim "$D" || exit 1
[ -f "$RUN/model_fp32.pt" ] || $PY -m relevance_explainer.tiny.train --data "$D" --out "$RUN" --tokenizer data/corpus/tokenizer.json \
  --init runs/v2e-nosim/model_fp32.pt --lr 1e-4 --epochs 1 --bs 32 --dropout 0.1 --verdict-weight 5 --select val_verdict_acc \
  --test-limit 3000 > "$RUN.train.log" 2>&1 || exit 1
echo "trained $(date -u +%H:%M)"
RUN_DIR="$RUN" SUF=_v3 scripts/eval_sets.sh
scripts/dpo_run.sh "$RUN" "$D" itemfirst_nosim_v3 > "$RUN.dpo.log" 2>&1
grep -E "pairs" "$RUN.dpo.log"
RUN_DIR="$RUN-dpo" SUF=_v3 scripts/eval_sets.sh
scripts/numguard_eval.sh "$RUN-dpo"
echo "ROUND-F-DONE $(date -u +%H:%M)"
