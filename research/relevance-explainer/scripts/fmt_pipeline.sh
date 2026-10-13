#!/usr/bin/env bash
# Output-format test on the 2,048-token base, in two parts that run at the same time.
#   Base question: arm A (verdict, explanation) on the full round G data, trained as soon as the base is ready (no
#     teacher needed), against runs/v2g-from-dapt-4ep (same data and recipe on the old base).
#   Format question: A, B (+ grade, tl;dr) and C (+ claims and references) on the same 250k-pair sample, with the
#     teacher's tl;dr and grade written for the sample first. The teacher then continues with the remaining rows, for a
#     final full-data model in the winning format.
# Steps:
#   1. wait for runs/base-2k; evaluate it at 1,024 and 2,048 tokens
#   2. resume the teachers on both GPUs (Gemma stays paused, so GPU 0 has room to train) and wait until both answer
#   3. start A on the full data (GPU 1)
#   4. sample, pilot (400 rows) and gate; the run stops when the gate fails
#   5. teacher: the sample, validation and test samples and the realistic test; then the remaining rows in the background
#   6. build the three sample arms; train A and B on GPU 0, C on GPU 1 (4 epochs, same recipe)
#   7. evaluate the four runs on the version-4 test sets and the Eurostat tables
# Every step skips when its output exists, so a restart resumes.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' LLM_REASONING=
PY="$HOME/venvs/relexp/bin/python"
LOGD=/data/wb536061/ai4data-relevance-explainer
SRC=data/v2_g_itemfirst_nosim
F=data/fmt          # full-data teacher output
S=data/fmt_s        # the sample and its arms
BASE=runs/base-2k/model.pt
COMMON="--tokenizer data/corpus_multi_final/tokenizer.json --init $BASE --lr 1e-3 --epochs 4 --bs 32 --dropout 0.1 \
  --verdict-weight 5 --select val_verdict_acc --max-len 2048 --max-new 420 --test-limit 3000"
up() { curl -s -o /dev/null --max-time 5 -w "%{http_code}" "localhost:$1/v1/models" | grep -q 200; }
train() {  # data dir, run dir, GPU, extra flags
  [ -f "$2/model_fp32.pt" ] || CUDA_VISIBLE_DEVICES=$3 $PY -m relevance_explainer.tiny.train --data "$1" --out "$2" $COMMON $4 > "$2.train.log" 2>&1
}
echo "fmt start $(date -u +%H:%M)"

# 1. the base
until [ -f runs/base-2k/done ]; do sleep 120; done
for L in 1024 2048; do
  [ -f "runs/base-2k/eval_full_$L.json" ] || CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_base.py "$BASE" data/corpus_multi_final \
    "runs/base-2k/eval_full_$L" --seq "$L" > "runs/base-2k/eval_full_$L.log" 2>&1 || exit 1
  echo "base 2k at $L tokens:"; grep -v -i warn "runs/base-2k/eval_full_$L.log"
done

# 2. teachers (pause_gemma stays)
touch "$LOGD/pause_gemma"
rm -f "$LOGD/pause_gpu0" "$LOGD/pause_gpu1"
until up 8000 && up 8001; do sleep 60; done
echo "teachers up $(date -u +%H:%M)"

# 3. A on the full data, in the background
train "$SRC" runs/fmt-A-full 1 "" & PFULL=$!
echo "A full-data started $(date -u +%H:%M)"

# 4. sample, pilot, gate
mkdir -p "$F" "$S/src"
[ -f "$F/src_test.jsonl" ] || $PY scripts/build_format_arms.py "$SRC" none "$F" --samples-only || exit 1
[ -f "$S/src/val.jsonl" ] || cp "$F/src_val.jsonl" "$S/src/val.jsonl"
[ -f "$S/src/test.jsonl" ] || cp "$F/src_test.jsonl" "$S/src/test.jsonl"
[ -f "$S/src/train.jsonl" ] || $PY scripts/sample_unique_rows.py "$SRC/train.jsonl" "$S/src/train.jsonl" --n 250000 | tee "$S/sample.txt"
[ -f "$S/pilot.jsonl" ] || $PY -m relevance_explainer.synth.tldr_grade --out "$S/pilot.jsonl" "$S/src/train.jsonl" --sample 400 \
  --seed 7 --workers 96 > "$S/pilot.log" 2>&1 || exit 1
tail -1 "$S/pilot.log"
$PY scripts/tldr_pilot_report.py "$S/pilot.jsonl" "$S/src/train.jsonl" | tee "$S/pilot_report.txt"
if ! grep -q "GATE PASSED" "$S/pilot_report.txt"; then echo "FMT-STOPPED pilot gate $(date -u +%H:%M)"; wait $PFULL; exit 1; fi

# 5. teacher: the sample first (the pilot's answers are reused), then the rest in the background
[ -f "$S/teacher.jsonl" ] || cp "$S/pilot.jsonl" "$S/teacher.jsonl"
if [ ! -f "$S/teacher.done" ]; then
  $PY -m relevance_explainer.synth.tldr_grade --out "$S/teacher.jsonl" "$S/src/train.jsonl" "$S/src/val.jsonl" "$S/src/test.jsonl" \
    data/eval_real/itemfirst_nosim_v4/test.jsonl --workers 128 >> "$S/teacher.log" 2>&1 || exit 1
  touch "$S/teacher.done"
fi
tail -1 "$S/teacher.log"
echo "teacher sample done $(date -u +%H:%M)"
if [ ! -f "$F/teacher_rest.done" ]; then
  ( $PY -m relevance_explainer.synth.tldr_grade --out "$F/teacher_rest.jsonl" "$SRC/train.jsonl" --skip "$S/teacher.jsonl" \
      --workers 128 >> "$F/teacher_rest.log" 2>&1 && touch "$F/teacher_rest.done" ) &
fi

# 6. the sample's arms
[ -f "$S/C/test.jsonl" ] || $PY scripts/build_format_arms.py "$S/src" "$S/teacher.jsonl" "$S" --workers 96 > "$S/build.log" 2>&1 || exit 1
tail -12 "$S/build.log"
train "$S/A" runs/fmt-s-A 0 "" & PA=$!
train "$S/B" runs/fmt-s-B 0 "--special-tokens" & PB=$!
train "$S/C" runs/fmt-s-C 1 "--special-tokens" & PC=$!
wait $PA $PB $PC
echo "sample arms trained $(date -u +%H:%M)"
wait $PFULL
echo "A full-data trained $(date -u +%H:%M)"

# 7. evaluate (GPU 1)
for R in runs/fmt-s-A runs/fmt-s-B runs/fmt-s-C runs/fmt-A-full; do
  [ -f "$R/model_fp32.pt" ] || { echo "missing $R"; continue; }
  RUN_DIR=$R SUF=_v4 scripts/eval_sets.sh
  [ -f "$R/tables_generations_int8.jsonl" ] || CUDA_VISIBLE_DEVICES=1 $PY -m relevance_explainer.tiny.evaluate --run "$R" \
    --data data/eval_tables/itemfirst_nosim --precisions int8 --tag tables --script-guard 2>&1 | grep -E "Error|int8 " | tail -2
done
echo "FMT-DONE $(date -u +%H:%M) (the teacher may still be writing the remaining rows: $F/teacher_rest.log)"
