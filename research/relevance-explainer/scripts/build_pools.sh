#!/usr/bin/env bash
# Build the record pools for tables, map layers and scripts. Usage: scripts/build_pools.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="$HOME/venvs/relexp/bin/python -m relevance_explainer.synth.pools"
[ -f data/pools/script_rrr.jsonl ] || $PY nada --out data/pools/script_rrr.jsonl --base https://reproducibility.worldbank.org --n 1000
[ -f data/pools/table_ilostat.jsonl ] || $PY ilostat --out data/pools/table_ilostat.jsonl
[ -f data/pools/table_statcan.jsonl ] || $PY statcan --out data/pools/table_statcan.jsonl --n 3000
[ -f data/pools/geospatial_hdx.jsonl ] || $PY hdxgeo --out data/pools/geospatial_hdx.jsonl --n 3000
wc -l data/pools/*.jsonl
echo POOLS-DONE
