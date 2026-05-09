#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/swebench_lite.yaml
RUN_ID=multibranch_$(date +%Y%m%d_%H%M%S)
OUT_DIR=data
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
python -m agentweaver.workloads.swebench_loader --config "$CONFIG" --out "$OUT_DIR/processed/${RUN_ID}_instances.json"
echo "Multi-branch SWE execution is framework-dependent. Convert generated .traj files with:"
echo "python -m agentweaver.tracing.swe_trace_adapter --traj run.traj --out $OUT_DIR/traces/$RUN_ID/instance_branch.jsonl"
