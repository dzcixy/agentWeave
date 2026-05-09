#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/swebench_lite.yaml
RUN_ID=${RUN_ID:-small}
OUT_DIR=data
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
python -m agentweaver.simulator.gpu_cache_sim --processed "$OUT_DIR/processed/$RUN_ID" --config "$CONFIG" --out "$OUT_DIR/results/gpu_cache_${RUN_ID}.csv"
