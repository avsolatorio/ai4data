#!/usr/bin/env bash
# Unseen realistic test set: searches on the reserved fresh themes (never used for training data), labelled by the
# teacher from the full format, plus the teacher's thinking mode and Gemma 4 for consensus labels.
# Usage: scripts/realistic_test.sh [out=data/real_test] [n_theme=700] [n_anchor=900]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-data/real_test}; NT=${2:-700}; NA=${3:-900}
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
export LLM_BASE_URL=${LLM_BASE_URL:-http://localhost:8000/v1,http://localhost:8001/v1}
export QUERY_MODEL=qwen3.8-27b NARRATE_MODEL=qwen3.8-27b TEACHER_MODEL=qwen3.8-27b LLM_REASONING=
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
PY="$HOME/venvs/relexp/bin/python -m relevance_explainer.synth.realistic"
[ -f "$OUT/queries.jsonl" ] || $PY queries --out "$OUT" --n "$NT" --seed 9001 --themes data/fresh/themes.json
[ -f "$OUT/anchored.jsonl" ] || $PY anchored --out "$OUT" --n "$NA" --seed 9001 --themes data/fresh/themes.json --retrieve-workers 16
[ -f "$OUT/pairs.jsonl" ] || $PY retrieve --out "$OUT" --seed 9001 --retrieve-workers 16
[ -f "$OUT/rendered.jsonl" ] || $PY render --out "$OUT"
[ -f "$OUT/examples.jsonl" ] || $PY narrate --out "$OUT"
"$HOME/venvs/relexp/bin/python" -m relevance_explainer.synth.label_consensus --data "$OUT"
echo REALISTIC-TEST-DONE
