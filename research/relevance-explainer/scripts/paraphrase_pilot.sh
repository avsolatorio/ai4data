#!/usr/bin/env bash
# Pilot of paraphrased fully-answering searches: anchored stage only, every 'full' search paraphrased.
# Usage: scripts/paraphrase_pilot.sh <out_dir> [n=40]
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 PARAPHRASE_SHARE=1
export LLM_BASE_URL=http://localhost:8000/v1,http://localhost:8001/v1 QUERY_MODEL=qwen3.8-27b NARRATE_MODEL=qwen3.8-27b LLM_REASONING=
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
mkdir -p "$1"
"$HOME/venvs/relexp/bin/python" -m relevance_explainer.synth.realistic anchored --out "$1" --n "${2:-40}" --seed 777 --themes data/themes.json --retrieve-workers 20 2>&1 | grep -v Warning | tail -2
"$HOME/venvs/relexp/bin/python" -c "
import json, sys
for l in open(sys.argv[1] + '/anchored.jsonl'):
    r = json.loads(l)
    if r['relation'] == 'full':
        print(r['wording'], '|', r['language'], '|', r['record']['name'][:60], '=>', r['query'])" "$1"
