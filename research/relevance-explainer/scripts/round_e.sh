#!/usr/bin/env bash
# Round E: render version 3 (thresholds, units, exclusions, named periods) and the concept rules.
# 1. Relabel the phenomenon slices at version 3; 2. concept inventory, searches and labels; 3. assemble with all targeted
# sets; 4. continue the round D model; 5. evaluate on the version-3 test sets; 6. DPO and evaluate again.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' LLM_REASONING=
export LLM_BASE_URL=http://localhost:8000/v1,http://localhost:8001/v1 QUERY_MODEL=qwen3.8-27b NARRATE_MODEL=qwen3.8-27b
PY="$HOME/venvs/relexp/bin/python"
CH="$PY -m relevance_explainer.synth.challenge"
echo "round E start $(date -u +%H:%M)"
# 1. phenomenon slices relabelled at version 3 with the concept rules
P=data/real_phen/p1v3
mkdir -p "$P"; [ -f "$P/searches.jsonl" ] || cp data/real_phen/p1/searches.jsonl "$P/searches.jsonl"
[ -f "$P/labeled.jsonl" ] || CUDA_VISIBLE_DEVICES=1 $CH label --out "$P" --workers 32 || exit 1
[ -f "$P/examples.jsonl" ] || $CH examples --out "$P" || exit 1
echo "slices done $(date -u +%H:%M)"
# 2. concept inventory
C=data/concepts
[ -f "$C/inventory.jsonl" ] || $PY -m relevance_explainer.synth.concepts inventory --out "$C" --n 2000 || exit 1
[ -f "$C/searches.jsonl" ] || $PY -m relevance_explainer.synth.concepts searches --out "$C" || exit 1
[ -f "$C/labeled.jsonl" ] || CUDA_VISIBLE_DEVICES=1 $CH label --out "$C" --workers 32 || exit 1
[ -f "$C/examples.jsonl" ] || $CH examples --out "$C" || exit 1
echo "concepts done $(date -u +%H:%M)"
# 3. assemble
export CUDA_VISIBLE_DEVICES=1
D=data/v2_e; DN=data/v2_e_itemfirst_nosim; RUN=runs/v2e-nosim
[ -f "${D}_itemfirst/train.jsonl" ] || $PY -m relevance_explainer.synth.assemble_v2 --out "$D" --old-share 0.4 \
  --extra "para=data/real_para/c*:2" "types=data/real_types/c*:2" "metatrans=data/real_meta/m*" "phen=$P:2" "concepts=$C:2" || exit 1
[ -f "$DN/train.jsonl" ] || $PY scripts/strip_similarity.py "${D}_itemfirst" "$DN" || exit 1
echo "assembled $(date -u +%H:%M)"
# 4. continue the round D supervised model
[ -f "$RUN/model_fp32.pt" ] || $PY -m relevance_explainer.tiny.train --data "$DN" --out "$RUN" --tokenizer data/corpus/tokenizer.json \
  --init runs/v2d-nosim/model_fp32.pt --lr 2e-4 --epochs 2 --bs 32 --dropout 0.1 --verdict-weight 5 --select val_verdict_acc \
  > "$RUN.train.log" 2>&1 || exit 1
echo "trained $(date -u +%H:%M)"
# 5. evaluate (version-3 test sets)
RUN_DIR="$RUN" SUF=_v3 scripts/eval_sets.sh
# 6. DPO on version-3 inputs, then evaluate and judge
scripts/dpo_run.sh "$RUN" "$DN" itemfirst_nosim_v3 > "$RUN.dpo.log" 2>&1
RUN_DIR="$RUN-dpo" SUF=_v3 scripts/eval_sets.sh
echo "ROUND-E-DONE $(date -u +%H:%M)"
