#!/usr/bin/env bash
# One feedback-loop experiment with local models.  run_loop.sh NAME TASK SEED BASE ROUND
#   NAME    prefix of the two run folders, $SMALL_AI_RUNS/NAME-targeted and NAME-control
#   TASK    task folder under src/ai4data/small_ai/tasks (isco, wb_topics, coicop)
#   BASE    requests of round 0, shared by the two arms;  ROUND  requests of each later round
#   TOL     optional: tolerance of the acceptance rule (default 0.03)
#   GEN0    optional: a folder with examples.jsonl and raw.jsonl of an earlier generation, used as round 0
# The targeted arm trains on GPU 0 and the control arm on GPU 1. Both arms are then trained with three seeds
# and scored on the report set, and baselines.sh asks the large models directly. Every step is skipped when its
# output exists, so the script can be started again after an interruption.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
KEY="${OPENROUTER_API_KEY:-}"
source "$HERE/env.sh"
unset OPENROUTER_API_KEY   # the orchestrator reads provider keys; the loop itself uses the local servers only
NAME=$1 TASK=$REPO/src/ai4data/small_ai/tasks/$2 SEED=$3 BASE=$4 ROUND=$5
T=$RUNS/$NAME-targeted C=$RUNS/$NAME-control; mkdir -p "$RUNS"
fail() { echo "== failed: $1 $(date -u +%T)"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "== start $(date -u +%T)"
I="--rounds 5 --baseline-requests $BASE --round-requests $ROUND --seed $SEED --tolerance ${TOL:-0.03} --price 0,0"
[ -f $T/run.json ] || H init $T --task $TASK --arm targeted $I > /dev/null
if [ -n "${GEN0:-}" ] && [ ! -d $T/rounds/r00/gen ]; then mkdir -p $T/rounds/r00 && cp -r $GEN0 $T/rounds/r00/gen; fi
if [ ! -f $T/rounds/r00/decision.json ]; then
  if [ -d $T/rounds/r00/gen ] && [ -n "${GEN0:-}" ]; then CUDA_VISIBLE_DEVICES=0 H finish-round $T 0; else CUDA_VISIBLE_DEVICES=0 H round $T; fi
fi
[ -f $T/rounds/r00/decision.json ] || fail "round 0"
echo "== round 0 $(date -u +%T)"
[ -f $C/run.json ] || H init $C --task $TASK --arm control $I --baseline-from $T > /dev/null
( export CUDA_VISIBLE_DEVICES=0
  [ -f $T/stop.json ] || $REPO/src/ai4data/small_ai/orchestrator/run.sh $T
  echo "== orchestrator exited $? $(date -u +%T)" ) > $RUNS/$NAME-targeted.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1
  for k in 1 2 3 4 5; do [ -f $C/rounds/r0$k/decision.json ] || H round $C; done
  echo "== control done $(date -u +%T)" ) > $RUNS/$NAME-control.log 2>&1 &
wait
[ -f $C/rounds/r05/decision.json ] || fail "control arm"
echo "== arms done $(date -u +%T)"
[ -f $T/final.json ] || CUDA_VISIBLE_DEVICES=0 H final $T --seeds 42,1,2 > /dev/null &
[ -f $C/final.json ] || CUDA_VISIBLE_DEVICES=1 H final $C --seeds 42,1,2 > /dev/null &
wait
[ -f $T/final_r00.json ] || CUDA_VISIBLE_DEVICES=0 H final $T --seeds 42,1,2 --rounds 0 --tag final_r00 > /dev/null
for f in $T/final.json $C/final.json $T/final_r00.json; do [ -f $f ] || fail "$f"; done
echo "== finals $(date -u +%T)"
OPENROUTER_API_KEY="$KEY" "$HERE/baselines.sh" "$2"
echo "== end $(date -u +%T)"
