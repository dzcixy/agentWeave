#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH=${MODEL_PATH:-/data2/model_zoo/Qwen2.5-Coder-7B-Instruct}
if [ ! -d "$MODEL_PATH" ]; then MODEL_PATH=${MODEL_PATH_FALLBACK:-Qwen/Qwen2.5-Coder-7B-Instruct}; fi
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port ${PORT:-30000}
