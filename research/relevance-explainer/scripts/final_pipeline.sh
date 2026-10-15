#!/usr/bin/env bash
# Final model in format C (verdict, grade, claims, tl;dr, explanation with references) on the full round G data.
# Steps, each skipped when its output exists:
#   1. wait for the teacher's tl;dr and grade for the remaining rows (data/fmt/teacher_rest.done)
#   2. judge the sample arms' realistic-test answers on the teachers, in the background (scripts/judge_decode.sh)
#   3. build the full-data arms from both teacher files (data/final; only C is trained)
#   4. train and evaluate runs/final-C on GPU 0 (scripts/fmt_train_one.sh: same recipe as the format test)
#   5. decoding test (no repeat ban, verifier-selected resampling), structure report, and the judge for final-C
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="$HOME/venvs/relexp/bin/python"
LOGD=/data/wb536061/ai4data-relevance-explainer
SRC=data/v2_g_itemfirst_nosim
F=data/final
R=runs/final-C
REAL=data/eval_real/itemfirst_nosim_v4
echo "final start $(date -u +%H:%M)"

until [ -f data/fmt/teacher_rest.done ]; do sleep 120; done
echo "teacher done $(date -u +%H:%M)"
scripts/judge_decode.sh runs/fmt-s-A runs/fmt-s-B runs/fmt-s-C > "$LOGD/judge_decode_arms.log" 2>&1 & PJ=$!

[ -f "$F/C/test.jsonl" ] || $PY scripts/build_format_arms.py "$SRC" data/fmt_s/teacher.jsonl,data/fmt/teacher_rest.jsonl "$F" \
  --workers 96 > "$F.build.log" 2>&1 || { echo "build failed: $F.build.log"; exit 1; }
tail -12 "$F.build.log"
echo "built $(date -u +%H:%M)"

scripts/fmt_train_one.sh "$F/C" "$R" 0 --special-tokens || exit 1
echo "trained and evaluated $(date -u +%H:%M)"

[ -f "$R/decode_test_itemfirst_nosim_v4.jsonl" ] || CUDA_VISIBLE_DEVICES=0 $PY scripts/fmt_decode_test.py "$R" "$REAL/test.jsonl" --k 6 2>&1 | grep -v -i warn
$PY scripts/fmt_structure_report.py "$R" --sets realtest_v4 test 2>&1 | grep -v -i warn
wait $PJ
scripts/judge_decode.sh "$R"
grep -E "^(teacher|student|fmt|final)" "$LOGD/judge_decode_arms.log"
echo "FINAL-DONE $(date -u +%H:%M)"
