#!/usr/bin/env bash
# Recompute coverage weights (data/real/coverage.json) whenever another realistic chunk finishes.
# Usage: scripts/coverage_watch.sh   (runs until stopped; logs to stdout)
cd "$(dirname "$0")/.."
. scripts/env.sh
export PYTHONPATH=src
last=-1
while true; do
  n=$(ls data/real/c*/examples.jsonl 2>/dev/null | wc -l)
  if [ "$n" != "$last" ] && [ "$n" -gt 0 ]; then
    echo "$(date -u +%H:%M) chunks with data: $n; updating coverage"
    "$RELEXP_PY" -m relevance_explainer.synth.coverage --root data/real --themes data/themes.json > data/real/coverage_report.txt 2>&1 \
      && grep -E "^(examples|countries_mentioned|themes_used|items_unique|verdict:|distinct_bigram_ratio)" data/real/coverage_report.txt
    last=$n
  fi
  sleep 120
done
