#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH=${MODEL_PATH:-/data2/model_zoo/Qwen/Qwen2.5-Coder-7B-Instruct}
if [ ! -d "$MODEL_PATH" ]; then MODEL_PATH=${MODEL_PATH_FALLBACK:-Qwen/Qwen2.5-Coder-7B-Instruct}; fi
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} vllm serve "$MODEL_PATH" \
  --served-model-name qwen-coder-7b \
  --host 0.0.0.0 \
  --port ${PORT:-8000} \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
