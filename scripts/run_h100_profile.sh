#!/usr/bin/env bash
set -euo pipefail
SERVER=http://localhost:8000/v1
MODEL=qwen-coder-7b
OUT_DIR=data
while [[ $# -gt 0 ]]; do
  case "$1" in
    --server) SERVER="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --config|--run-id) shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
python -m agentweaver.profiling.profile_vllm --server "$SERVER" --model "$MODEL" --out "$OUT_DIR/profiles/h100_profile_raw.csv"
python -m agentweaver.profiling.fit_latency_model --raw "$OUT_DIR/profiles/h100_profile_raw.csv" --out "$OUT_DIR/profiles/h100_latency_model.json" --plot-dir "$OUT_DIR/plots"
