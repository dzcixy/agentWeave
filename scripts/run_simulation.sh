#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/wafer_6x6.yaml
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
for POLICY in naive_wafer acd_only acd_bes acd_nisp full_agentweaver; do
  python -m agentweaver.simulator.replay --processed "$OUT_DIR/processed/$RUN_ID" --wafer-config "$CONFIG" --policy "$POLICY" --out "$OUT_DIR/results/replay_${POLICY}_${RUN_ID}.csv"
done
