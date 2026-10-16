#!/usr/bin/env bash
# Re-run the render stage of one chunk (for example after a failure). Usage: scripts/render_chunk.sh <chunk_dir>
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
"$RELEXP_PY" -m relevance_explainer.synth.realistic render --out "$1" 2>&1 | grep -v Warning | tail -8
