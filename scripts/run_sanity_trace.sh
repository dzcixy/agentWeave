#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/swebench_lite.yaml
RUN_ID=sanity_$(date +%Y%m%d_%H%M%S)
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
echo "If SERVER_URL is set, run mini/SWE-agent externally and convert trajectories with swe_trace_adapter.py."
echo "Falling back to synthetic small trace path for DAG/characterization."
bash scripts/run_all_small.sh --config configs/small_sanity.yaml --run-id "$RUN_ID" --out-dir "$OUT_DIR"
