#!/usr/bin/env bash
# Metadata-translation sets m1..mN, each 3,000 pairs drawn from five realistic chunks (c41 onward, after the anchored
# language fix took effect from about c43). Stop with data/real_meta/STOP. Usage: scripts/meta_long.sh [sets=8]
set -uo pipefail
cd "$(dirname "$0")/.."
N=${1:-8}
for i in $(seq 1 "$N"); do
  [ -f data/real_meta/STOP ] && { echo "stop file found"; break; }
  OUT=data/real_meta/m$i
  [ -f "$OUT/examples.jsonl" ] && continue
  a=$((36 + 5 * i)); srcs=""
  for c in $(seq "$a" $((a + 4))); do [ -f "data/real/c$c/examples.jsonl" ] && srcs="$srcs data/real/c$c"; done
  [ -z "$srcs" ] && { echo "no source chunks for m$i"; continue; }
  echo "meta set $i start $(date -u +%H:%M) from$srcs"
  scripts/meta_translate.sh "$OUT" 3000 "$i" $srcs
done
echo META-LONG-DONE
