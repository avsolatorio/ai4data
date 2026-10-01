#!/usr/bin/env bash
# Serve the 8-bit Qwen3.8-27B teacher with vLLM (OpenAI-compatible API on port 8000).
# Usage: GPU=0 MEM=0.55 scripts/serve_vllm.sh
set -euo pipefail

MODEL_DIR=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/* | head -n 1)
export CUDA_VISIBLE_DEVICES=${GPU:-0}
# DeepGEMM JIT-compiles kernels and needs nvcc >= 12.9, which is not installed here.
export VLLM_USE_DEEP_GEMM=0
# The FlashInfer sampler also JIT-compiles CUDA kernels (nvcc and ninja); use the PyTorch sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

exec "$HOME/venvs/vllm/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name qwen3.8-27b \
  --port "${PORT:-8000}" \
  --max-model-len 4096 \
  --gpu-memory-utilization "${MEM:-0.55}" \
  --max-num-seqs "${SEQS:-128}" \
  --enable-prefix-caching \
  --language-model-only
