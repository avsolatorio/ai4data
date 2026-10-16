#!/usr/bin/env bash
# Round G: verdict rules (NARRATE_V3) and render version 4 for the targeted slices, lexical hard negatives, more
# exclusion / look-alike / proxy / group / period searches, and Eurostat tables (one half for training searches, the
# other half a held-out table test). Every step skips when its output exists, so a restart resumes.
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' LLM_REASONING= CUDA_VISIBLE_DEVICES=1
PY="$RELEXP_PY"
CH="$PY -m relevance_explainer.synth.challenge"
LAB="--prompt v3 --render-version 4 --workers 64"
echo "round G start $(date -u +%H:%M)"

label() {  # label + examples in a directory that has searches.jsonl
  [ -f "$1/labeled.jsonl" ] || $CH label --out "$1" $LAB || return 1
  [ -f "$1/examples.jsonl" ] || $CH examples --out "$1" || return 1
  echo "labelled $1 $(date -u +%H:%M)"
}

# 1. relabel the phenomenon and concept slices with the rules
for pair in "data/real_phen/p1 data/real_phen/p1v4" "data/concepts data/concepts_v4"; do
  set -- $pair
  mkdir -p "$2"; [ -f "$2/searches.jsonl" ] || cp "$1/searches.jsonl" "$2/searches.jsonl"
  label "$2" || exit 1
done
# 2. lexical hard negatives
[ -f data/real_hardneg/h1/searches.jsonl ] || $PY -m relevance_explainer.synth.hardneg --out data/real_hardneg/h1 --n 8000 || exit 1
label data/real_hardneg/h1 || exit 1
# 3. more searches for the weakest phenomena (training writers, records outside the held-out suite)
P2=data/real_phen/p2
[ -f "$P2/searches.jsonl" ] || $CH write --out "$P2" --per 1200 --phenomena negation,lookalike,proxy,group_definition,time_phrase \
  --exclude data/challenge/searches.jsonl --mix-writers --seed 41 --workers 64 || exit 1
label "$P2" || exit 1
# 4. Eurostat tables: half 0 for training searches, half 1 for the held-out table test (Gemma 4 writes, as in the suite)
ES=data/pools/table_eurostat.jsonl
until [ -s "$ES" ]; do sleep 60; done
TPH=synonym,standard_measure,threshold,finer_level,question,title_name,group_definition,lookalike,vague,multi_ask,negation,proxy,time_phrase
ET=data/real_estat/e1
[ -f "$ET/searches.jsonl" ] || $CH write --out "$ET" --records "$ES" --half 0 --per 250 --phenomena "$TPH" --mix-writers --seed 43 --workers 64 || exit 1
label "$ET" || exit 1
EV=data/estat_test
[ -f "$EV/searches.jsonl" ] || $CH write --out "$EV" --records "$ES" --half 1 --per 50 --phenomena "$TPH" --seed 47 --workers 64 || exit 1
[ -f "$EV/labeled.jsonl" ] || $CH label --out "$EV" $LAB || exit 1
[ -f data/eval_tables/itemfirst_nosim/test.jsonl ] || $CH build --out "$EV" --eval data/eval_tables || exit 1
# challenge suite relabelled with the rules (scripts/policy_relabel.sh), as the version-4 test copy
if [ ! -f data/eval_challenge/itemfirst_nosim_v4/test.jsonl ]; then
  $CH build --out data/challenge --labeled labeled_v4.jsonl --eval data/eval_challenge_rules || exit 1
  cp -r data/eval_challenge_rules/itemfirst_nosim data/eval_challenge/itemfirst_nosim_v4
fi
echo "data done $(date -u +%H:%M)"

# 5. assemble, strip the similarity line, drop narratives with invented numbers
D=data/v2_g; DN=data/v2_g_itemfirst_nosim; RUN=runs/v2g-nosim
[ -f "${D}_itemfirst/train.jsonl" ] || $PY -m relevance_explainer.synth.assemble_v2 --out "$D" --old-share 0.4 --drop-invented-numbers \
  --extra "para=data/real_para/c*:2" "types=data/real_types/c*:2" "metatrans=data/real_meta/m*" "phen=data/real_phen/p1v4:2" \
          "concepts=data/concepts_v4:2" "hardneg=data/real_hardneg/h1" "phen2=$P2:2" "estat=$ET:2" || exit 1
[ -f "$DN/train.jsonl" ] || $PY scripts/strip_similarity.py "${D}_itemfirst" "$DN" || exit 1
echo "assembled $(date -u +%H:%M)"
# 6. continue the round F supervised model
[ -f "$RUN/model_fp32.pt" ] || $PY -m relevance_explainer.tiny.train --data "$DN" --out "$RUN" --tokenizer data/corpus/tokenizer.json \
  --init runs/v2f-nosim/model_fp32.pt --lr 1e-4 --epochs 1 --bs 32 --dropout 0.1 --verdict-weight 5 --select val_verdict_acc \
  --test-limit 3000 > "$RUN.train.log" 2>&1 || exit 1
echo "trained $(date -u +%H:%M)"
# 7. evaluate, DPO, evaluate again; the table test for the new runs and the round F baseline
evals() {
  RUN_DIR="$1" SUF=_v4 scripts/eval_sets.sh
  [ -f "$1/tables_generations_int8.jsonl" ] || $PY -m relevance_explainer.tiny.evaluate --run "$1" --data data/eval_tables/itemfirst_nosim \
    --precisions int8 --tag tables --script-guard 2>&1 | grep -E "Error|int8 " | tail -2
  echo "== tables $1"; $PY -m relevance_explainer.synth.score_real --data data/eval_tables/itemfirst_nosim --run "$1" --tag tables | grep -E "teacher|unanimous|by_mode"
}
evals runs/v2f-nosim-dpo
evals "$RUN"
[ -f "$RUN-dpo/model_fp32.pt" ] || scripts/dpo_run.sh "$RUN" "$DN" itemfirst_nosim_v4 > "$RUN.dpo.log" 2>&1
grep -E "pairs" "$RUN.dpo.log"
evals "$RUN-dpo"
echo "ROUND-G-DONE $(date -u +%H:%M)"
