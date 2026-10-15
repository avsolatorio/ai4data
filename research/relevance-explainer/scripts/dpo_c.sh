#!/usr/bin/env bash
# One on-policy DPO round for a format-C run (distill/onpolicy_c.py), then evaluation: test sets (eval_sets.sh),
# decoding test with retries, structure report and the judge of the realistic-test answers.
#   scripts/dpo_c.sh <run> <data dir> [n inputs] [k samples] [GPU]
#   scripts/dpo_c.sh runs/final-C data/final/C 15000 8 0
# Steps skip when their output exists. The teacher servers run only for scoring and judging (scripts/teachers.sh):
# sampling, DPO training and the test sets have the GPUs without them.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${1:?run}; D=${2:?data}; N=${3:-15000}; K=${4:-8}; GPU=${5:-0}
export PYTHONPATH=src TEACHER_URLS=http://localhost:8000/v1,http://localhost:8001/v1
PY="$HOME/venvs/relexp/bin/python"
TAG=$(basename "$RUN")
OUT=$RUN-dpo
S="onpolicy_c_${TAG}_samples.jsonl"; SC="onpolicy_c_${TAG}_scored.jsonl"; P="dpo_c_pairs_${TAG}.jsonl"
[ -f "$D/$S" ] || CUDA_VISIBLE_DEVICES=$GPU $PY -m relevance_explainer.distill.onpolicy_c sample --run "$RUN" --data "$D" --n "$N" --k "$K" --out "$S" || exit 1
if [ ! -f "$D/$SC" ]; then
  scripts/teachers.sh up || exit 1
  $PY -m relevance_explainer.distill.onpolicy_c score --data "$D" --samples "$S" --out "$SC" || { scripts/teachers.sh down; exit 1; }
  scripts/teachers.sh down
fi
[ -f "$D/$P" ] || $PY -m relevance_explainer.distill.onpolicy_c pairs --data "$D" --scored "$SC" --out "$P" || exit 1
[ -f "$OUT/model_fp32.pt" ] || CUDA_VISIBLE_DEVICES=$GPU $PY -m relevance_explainer.tiny.dpo --run "$RUN" --data "$D" --pairs "$P" --out "$OUT" \
  --beta 0.3 --lr 2e-5 --epochs 2 --bs 16 --sft 0.5 --max-len 2048 || exit 1
RUN_DIR=$OUT SUF=_v4 scripts/eval_sets.sh  # GPU 1
[ -f "$OUT/decode_test_itemfirst_nosim_v4.jsonl" ] || CUDA_VISIBLE_DEVICES=$GPU $PY scripts/fmt_decode_test.py "$OUT" \
  data/eval_real/itemfirst_nosim_v4/test.jsonl --k 6 2>&1 | grep -v -i warn
if [ ! -f "$OUT/judge_tldr_realnb.json" ]; then
  scripts/teachers.sh up || exit 1
  scripts/judge_decode.sh "$OUT"
  scripts/teachers.sh down
fi
echo "DPO-C-DONE $OUT $(date -u +%H:%M)"
