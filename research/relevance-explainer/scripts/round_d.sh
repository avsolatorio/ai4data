#!/usr/bin/env bash
# Round D: all realistic chunks plus the targeted sets (paraphrase x2, new item types x2, translated metadata), no score
# line; continue the no-score model, evaluate on every test set, then on-policy DPO and evaluate again.
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
PY="$RELEXP_PY"
D=data/v2_d; DN=data/v2_d_itemfirst_nosim; RUN=runs/v2d-nosim
echo "round D start $(date -u +%H:%M)"
[ -f "${D}_itemfirst/train.jsonl" ] || $PY -m relevance_explainer.synth.assemble_v2 --out "$D" --old-share 0.4 \
  --extra "para=data/real_para/c*:2" "types=data/real_types/c*:2" "metatrans=data/real_meta/m*" || exit 1
[ -f "$DN/train.jsonl" ] || $PY scripts/strip_similarity.py "${D}_itemfirst" "$DN" || exit 1
[ -f data/schemas_eval2/itemfirst_nosim/test.jsonl ] || $PY scripts/strip_similarity.py data/schemas_eval2/itemfirst data/schemas_eval2/itemfirst_nosim || exit 1
echo "assembled $(date -u +%H:%M)"
evals() {  # run tag
  for pair in "data/eval_real/itemfirst_nosim realtest" "data/eval_para/itemfirst_nosim para" "data/schemas_eval2/itemfirst_nosim schemas"; do
    set -- $pair "$1"
    $PY -m relevance_explainer.tiny.evaluate --run "$3" --data "$1" --precisions int8 --tag "$2" --script-guard > /dev/null 2>&1
    echo "$2: $($PY -m relevance_explainer.synth.score_real --data "$1" --run "$3" --tag "$2" | head -1)"
  done
  scripts/judge_tag.sh "$1" realtest 2>&1 | grep -E "^student"
}
[ -f "$RUN/model_fp32.pt" ] || $PY -m relevance_explainer.tiny.train --data "$DN" --out "$RUN" --tokenizer data/corpus/tokenizer.json \
  --init runs/v2c-cont-itemfirst-nosim/model_fp32.pt --lr 2e-4 --epochs 2 --bs 32 --dropout 0.1 --verdict-weight 5 --select val_verdict_acc \
  > "$RUN.train.log" 2>&1 || exit 1
echo "trained $(date -u +%H:%M)"
evals "$RUN"
scripts/dpo_run.sh "$RUN" "$DN" itemfirst_nosim > "$RUN.dpo.log" 2>&1
evals "$RUN-dpo"
echo "ROUND-D-DONE $(date -u +%H:%M)"
