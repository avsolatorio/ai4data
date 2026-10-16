#!/usr/bin/env bash
# Dry run of the render stage on the first 20 pairs of a chunk. Usage: scripts/render_dry.sh <chunk_dir> <out_dir>
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
mkdir -p "$2"
head -20 "$1/pairs.jsonl" > "$2/pairs.jsonl"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1
"$RELEXP_PY" -m relevance_explainer.synth.realistic render --out "$2" 2>&1 | grep -v Warning | tail -2
"$RELEXP_PY" -c "
import json, sys
r = [json.loads(l) for l in open(sys.argv[1] + '/rendered.jsonl')]
print('versions', set(x['render_version'] for x in r))
print(r[0]['input_v2'][-600:])" "$2"
