#!/usr/bin/env bash
# Rerun the deterministic steps behind runs/final-C-dpo and compare the outputs with the released artifacts
# (artifacts.json, REPRODUCE.md). Writes under data/repro and runs/repro-*, never over released files.
#   1. format C data: scripts/build_format_arms.py twice, with Python hash seeds 1 and 2 (CPU)
#   2. calibration: scripts/calibrate_c.py (CPU)
#   3. faithfulness scorer: scripts/faith_scorer.py with its features recomputed (GPU)
#   4. evaluation: the realistic test generations of the released model (GPU)
#   5. DPO training from the released pairs, then its realistic-test accuracy (GPU; kernels are not bit-deterministic)
#   scripts/verify_reproduction.sh [GPU=0]
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
PY="$RELEXP_PY"
GPU=${1:-0}
R=runs/final-C-dpo; D=data/final/C; REAL=data/eval_real/itemfirst_nosim_v4
mkdir -p data/repro runs/repro-final-C-dpo
same() {  # label, file a, file b
  if cmp -s "$2" "$3"; then echo "SAME      $1"; else echo "DIFFERENT $1 ($2 vs $3)"; fi
}

echo "== 1. format C data"
for h in 1 2; do
  O=data/repro/final_h$h
  [ -f "$O/C/test.jsonl" ] || PYTHONHASHSEED=$h $PY scripts/build_format_arms.py data/v2_g_itemfirst_nosim \
    data/fmt_s/teacher.jsonl,data/fmt/teacher_rest.jsonl "$O" --workers 96 > "$O.log" 2>&1 || { echo "build failed: $O.log"; exit 1; }
  for s in train val test; do same "C/$s.jsonl, hash seed $h" "$O/C/$s.jsonl" "$D/$s.jsonl"; done
done

echo "== 2. calibration"
[ -f data/repro/calibration.json ] || $PY scripts/calibrate_c.py "$R" "$D/val.jsonl" "$REAL/test.jsonl" --out data/repro/calibration.json > /dev/null 2>&1
same "calibration.json" data/repro/calibration.json "$R/calibration.json"

echo "== 3. faithfulness scorer (features recomputed)"
[ -f data/repro/faith_scorer.json ] || CUDA_VISIBLE_DEVICES=$GPU $PY scripts/faith_scorer.py "$R" "$D" faith_final-C-dpo_s1_scored.jsonl \
  --batch 32 --out data/repro/faith_scorer.json --cache data/repro/faith_features.npz 2>&1 | grep -E '^\| combined'
same "faith_scorer.json" data/repro/faith_scorer.json "$R/faith_scorer.json"
$PY - <<'EOF'
import json
a, b = (json.load(open(p)) for p in ("data/repro/faith_scorer.json", "runs/final-C-dpo/faith_scorer.json"))
d = max(abs(x - y) for x, y in zip(a["coef"], b["coef"]))
print(f"          largest coefficient difference {d:.2e}; combined AUROC {a['results']['combined']['auroc']:.4f} "
      f"against {b['results']['combined']['auroc']:.4f}")
EOF

echo "== 4. evaluation of the released model"
[ -f "$R/realtest_v4_repro_generations_int8.jsonl" ] || CUDA_VISIBLE_DEVICES=$GPU $PY -m relevance_explainer.tiny.evaluate --run "$R" \
  --data "$REAL" --precisions int8 --tag realtest_v4_repro --script-guard 2>&1 | grep -E "^int8" | tail -1
same "realistic-test generations" "$R/realtest_v4_repro_generations_int8.jsonl" "$R/realtest_v4_generations_int8.jsonl"

echo "== 5. DPO training from the released pairs"
O=runs/repro-final-C-dpo
[ -f "$O/model_fp32.pt" ] || CUDA_VISIBLE_DEVICES=$GPU $PY -m relevance_explainer.tiny.dpo --run runs/final-C --data "$D" \
  --pairs dpo_c_pairs_final-C.jsonl --out "$O" --beta 0.3 --lr 2e-5 --epochs 2 --bs 16 --sft 0.5 --max-len 2048 > "$O.log" 2>&1 \
  || { echo "DPO failed: $O.log"; exit 1; }
same "DPO model weights" "$O/model_fp32.pt" "$R/model_fp32.pt"
[ -f "$O/realtest_v4_generations_int8.jsonl" ] || CUDA_VISIBLE_DEVICES=$GPU $PY -m relevance_explainer.tiny.evaluate --run "$O" \
  --data "$REAL" --precisions int8 --tag realtest_v4 --script-guard 2>&1 | grep -E "^int8" | tail -1
$PY - <<'EOF'
import json
for r in ("runs/final-C-dpo", "runs/repro-final-C-dpo"):
    m = json.load(open(f"{r}/realtest_v4_metrics.json"))
    m = m.get("int8", m.get("precision", {}).get("int8", m))
    print(f"          {r}: realistic-test verdict accuracy {100 * m['verdict_accuracy']:.2f}")
EOF
echo "VERIFY-DONE $(date -u +%H:%M)"
