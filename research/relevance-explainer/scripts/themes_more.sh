#!/usr/bin/env bash
# Expand search themes with Qwen and Gemma 4 (see synth/themes_more.py). Usage: scripts/themes_more.sh [per=40]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export LLM_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
exec "$HOME/venvs/relexp/bin/python" -m relevance_explainer.synth.themes_more --per "${1:-40}"
