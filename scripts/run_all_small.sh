#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/small_sanity.yaml
RUN_ID=small_$(date +%Y%m%d_%H%M%S)
OUT_DIR=data
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
TRACE_DIR="$OUT_DIR/traces/$RUN_ID"
PROCESSED="$OUT_DIR/processed/$RUN_ID"
RESULTS="$OUT_DIR/results"
PLOTS="$OUT_DIR/plots"
mkdir -p "$TRACE_DIR" "$PROCESSED" "$RESULTS" "$PLOTS"
python -m agentweaver.workloads.synthetic_fork_join --config "$CONFIG" --out-dir "$OUT_DIR/traces" --run-id "$RUN_ID"
python -m agentweaver.analysis.context_segment_graph --trace-dir "$TRACE_DIR" --out "$PROCESSED" --config configs/default.yaml
python -m agentweaver.simulator.gpu_cache_sim --processed "$PROCESSED" --config "$CONFIG" --out "$RESULTS/gpu_cache_${RUN_ID}.csv"
python -m agentweaver.simulator.acd_mapping --processed "$PROCESSED" --config configs/wafer_4x4.yaml --out "$RESULTS/acd_mapping_${RUN_ID}.csv"
for POLICY in naive_wafer acd_only acd_bes acd_nisp full_agentweaver; do
  python -m agentweaver.simulator.replay --processed "$PROCESSED" --wafer-config configs/wafer_4x4.yaml --policy "$POLICY" --out "$RESULTS/replay_${POLICY}_${RUN_ID}.csv"
done
cp "$RESULTS/replay_full_agentweaver_${RUN_ID}.csv" "$RESULTS/ablation_${RUN_ID}.csv"
python -m agentweaver.plotting.plot_all --results-dir "$RESULTS" --out-dir "$PLOTS"
echo "Synthetic small pipeline complete"
echo "run_id=$RUN_ID"
echo "processed=$PROCESSED"
echo "results=$RESULTS"
echo "plots=$PLOTS"
python - <<'PY'
import csv, glob
for path in sorted(glob.glob("data/results/replay_*_*.csv"))[-5:]:
    with open(path, newline="") as f:
        rows=list(csv.DictReader(f))
    agg=rows[-1] if rows else {}
    print(path, "policy=", agg.get("policy"), "jct=", agg.get("jct"), "prefill_avoided=", agg.get("prefill_tokens_avoided"))
PY
