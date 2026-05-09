#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/synthetic.yaml
RUN_ID=synthetic_$(date +%Y%m%d_%H%M%S)
OUT_DIR=data
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
python -m agentweaver.workloads.synthetic_fork_join --config "$CONFIG" --out-dir "$OUT_DIR/traces" --run-id "$RUN_ID"
