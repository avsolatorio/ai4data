#!/usr/bin/env bash
# Round E training slices for the challenge phenomena: training writers, records outside the held-out suite, rows kept
# where the teacher and Gemma 4 agree. Usage: scripts/phen_train.sh [out=data/real_phen/p1] [seed=21]
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
OUT=${1:-data/real_phen/p1}; SEED=${2:-21}
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' LLM_REASONING=
PY="$RELEXP_PY -m relevance_explainer.synth.challenge"
PER='{"synonym": 1500, "threshold": 1500, "proxy": 1500, "time_phrase": 1500, "group_definition": 1500, "negation": 1500, "lookalike": 1500, "standard_measure": 1500}'
[ -f "$OUT/searches.jsonl" ] || $PY write --out "$OUT" --per 200 --per-phenomenon "$PER" --exclude data/challenge/searches.jsonl --mix-writers --seed "$SEED" --workers 32 || exit 1
[ -f "$OUT/labeled.jsonl" ] || $PY label --out "$OUT" --workers 32 || exit 1
$PY examples --out "$OUT"
echo PHEN-TRAIN-DONE
