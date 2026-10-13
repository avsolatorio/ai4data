#!/usr/bin/env bash
# Output-format test on the 2,048-token base: A (verdict, explanation), B (+ grade, tl;dr), C (+ claims and references).
#   1. wait for runs/base-2k; evaluate it at 1,024 and 2,048 tokens
#   2. resume the teachers (delete the pause files; the keepalive starts them) and wait until both answer
#   3. pilot: tl;dr and grade for 400 training rows; stop unless the gate passes (scripts/tldr_pilot_report.py)
#   4. tl;dr and grade for every unique training row, the validation and test samples and the realistic test
#   5. build the three arms (same rows in each)
#   6. free GPU 0 (pause and stop its servers), train A and B on GPU 0 and C on GPU 1 (4 epochs each, same recipe)
#   7. evaluate the three runs on the version-4 test sets and the Eurostat tables
# Judging and the tl;dr, claim and grade analyses follow by hand. Every step skips when its output exists.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' LLM_REASONING=
PY="$HOME/venvs/relexp/bin/python"
LOGD=/data/wb536061/ai4data-relevance-explainer
SRC=data/v2_g_itemfirst_nosim
F=data/fmt
BASE=runs/base-2k/model.pt
up() { curl -s -o /dev/null --max-time 5 -w "%{http_code}" "localhost:$1/v1/models" | grep -q 200; }
echo "fmt start $(date -u +%H:%M)"

# 1. the base
until [ -f runs/base-2k/done ]; do sleep 120; done
for S in 1024 2048; do
  [ -f "runs/base-2k/eval_full_$S.json" ] || CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_base.py "$BASE" data/corpus_multi_final \
    "runs/base-2k/eval_full_$S" --seq "$S" > "runs/base-2k/eval_full_$S.log" 2>&1 || exit 1
  echo "base 2k at $S tokens:"; grep -v -i warn "runs/base-2k/eval_full_$S.log"
done

# 2. teachers
rm -f "$LOGD/pause_gpu0" "$LOGD/pause_gpu1"
until up 8000 && up 8001; do sleep 60; done
echo "teachers up $(date -u +%H:%M)"

# 3. pilot
mkdir -p "$F"
[ -f "$F/src_test.jsonl" ] || $PY scripts/build_format_arms.py "$SRC" none "$F" --samples-only || exit 1
[ -f "$F/pilot.jsonl" ] || $PY -m relevance_explainer.synth.tldr_grade --out "$F/pilot.jsonl" "$SRC/train.jsonl" --sample 400 \
  --seed 7 --workers 96 > "$F/pilot.log" 2>&1 || exit 1
$PY scripts/tldr_pilot_report.py "$F/pilot.jsonl" "$SRC/train.jsonl" | tee "$F/pilot_report.txt"
if ! grep -q "GATE PASSED" "$F/pilot_report.txt"; then echo "FMT-STOPPED pilot gate $(date -u +%H:%M)"; exit 1; fi

# 4. all rows (the pilot's answers are reused: same keys)
[ -f "$F/teacher.jsonl" ] || cp "$F/pilot.jsonl" "$F/teacher.jsonl"
if [ ! -f "$F/teacher.done" ]; then
  $PY -m relevance_explainer.synth.tldr_grade --out "$F/teacher.jsonl" "$SRC/train.jsonl" "$F/src_val.jsonl" "$F/src_test.jsonl" \
    data/eval_real/itemfirst_nosim_v4/test.jsonl --workers 128 >> "$F/teacher.log" 2>&1 || exit 1
  touch "$F/teacher.done"
fi
tail -1 "$F/teacher.log"
echo "teacher done $(date -u +%H:%M)"

# 5. arms
[ -f "$F/C/test.jsonl" ] || $PY scripts/build_format_arms.py "$SRC" "$F/teacher.jsonl" "$F" --workers 96 > "$F/build.log" 2>&1 || exit 1
tail -6 "$F/build.log"

# 6. train; GPU 0's servers are paused for the training and resumed for judging afterwards
touch "$LOGD/pause_gpu0"
pkill -u "$(id -u)" -f "vllm serve.*--port 8000"; pkill -u "$(id -u)" -f "vllm serve.*--port 8002"
sleep 60
COMMON="--tokenizer data/corpus_multi_final/tokenizer.json --init $BASE --lr 1e-3 --epochs 4 --bs 32 --dropout 0.1 \
  --verdict-weight 5 --select val_verdict_acc --max-len 2048 --max-new 420 --test-limit 3000"
train() {  # arm, GPU, extra flags
  local R=runs/fmt-$1
  [ -f "$R/model_fp32.pt" ] || CUDA_VISIBLE_DEVICES=$2 $PY -m relevance_explainer.tiny.train --data "$F/$1" --out "$R" $COMMON $3 \
    > "$R.train.log" 2>&1
}
train A 0 "" & PA=$!
train B 0 "--special-tokens" & PB=$!
train C 1 "--special-tokens" & PC=$!
wait $PA $PB $PC
for X in A B C; do [ -f "runs/fmt-$X/model_fp32.pt" ] || { echo "FMT-STOPPED training $X failed $(date -u +%H:%M)"; exit 1; }; done
echo "trained $(date -u +%H:%M)"

# 7. evaluate (GPU 1, next to its teacher, as before)
for X in A B C; do
  RUN_DIR=runs/fmt-$X SUF=_v4 scripts/eval_sets.sh
  [ -f "runs/fmt-$X/tables_generations_int8.jsonl" ] || CUDA_VISIBLE_DEVICES=1 $PY -m relevance_explainer.tiny.evaluate --run "runs/fmt-$X" \
    --data data/eval_tables/itemfirst_nosim --precisions int8 --tag tables --script-guard 2>&1 | grep -E "Error|int8 " | tail -2
done
rm -f "$LOGD/pause_gpu0"  # the teacher and Gemma on GPU 0 come back for judging
echo "FMT-DONE $(date -u +%H:%M)"
