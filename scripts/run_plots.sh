#!/usr/bin/env bash
set -euo pipefail
OUT_DIR=data
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --config|--run-id) shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
python -m agentweaver.plotting.plot_all --results-dir "$OUT_DIR/results" --out-dir "$OUT_DIR/plots"
