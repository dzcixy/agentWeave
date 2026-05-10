#!/usr/bin/env bash
set -euo pipefail

RUN_ID=h100calib_pr2_v2_1
TRACE_DIR=data/traces/synthetic
PROCESSED_DIR=data/processed/synthetic
RESULTS_DIR=data/results
PLOTS_DIR=data/plots
MODEL_JSON=data/profiles/h100_latency_model_pr2_v2.json
CONFIG=configs/small_sanity.yaml
REGEN_TRACES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --trace-dir) TRACE_DIR="$2"; shift 2 ;;
    --processed-dir) PROCESSED_DIR="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --plots-dir) PLOTS_DIR="$2"; shift 2 ;;
    --model-json) MODEL_JSON="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --regen-traces) REGEN_TRACES=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

if [ ! -f "$MODEL_JSON" ]; then
  echo "missing H100 latency model: $MODEL_JSON" >&2
  exit 1
fi

if [ -n "${PYTHON_CMD:-}" ]; then
  # shellcheck disable=SC2206
  PYTHON=($PYTHON_CMD)
else
  PYTHON=(uv run python)
fi

TMP_DIR="$RESULTS_DIR/.tmp_${RUN_ID}"
export TRACE_DIR PROCESSED_DIR RESULTS_DIR PLOTS_DIR TMP_DIR RUN_ID CONFIG
mkdir -p "$TRACE_DIR" "$PROCESSED_DIR" "$RESULTS_DIR" "$PLOTS_DIR" "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

if [ "$REGEN_TRACES" = "1" ] || [ ! -f "$PROCESSED_DIR/events.jsonl" ]; then
  "${PYTHON[@]}" -m agentweaver.workloads.synthetic_fork_join --fixed-scenarios --out-dir "$TRACE_DIR"
  "${PYTHON[@]}" -m agentweaver.analysis.context_segment_graph \
    --trace-dir "$TRACE_DIR" \
    --out "$PROCESSED_DIR" \
    --config configs/default.yaml
fi

cp "$MODEL_JSON" "$PROCESSED_DIR/h100_latency_model.json"

"${PYTHON[@]}" -m agentweaver.simulator.acd_mapping \
  --processed "$PROCESSED_DIR" \
  --config configs/wafer_4x4.yaml \
  --out "$RESULTS_DIR/acd_mapping_h100calib_pr2_v2_1.csv"

"${PYTHON[@]}" -m agentweaver.simulator.gpu_cache_sim \
  --processed "$PROCESSED_DIR" \
  --config "$CONFIG" \
  --out "$RESULTS_DIR/gpu_cache_baselines_h100calib_pr2_v2_1.csv"

POLICIES=(naive_wafer static_branch_pinning wafer_fcfs acd_only acd_bes acd_nisp full_agentweaver)
for POLICY in "${POLICIES[@]}"; do
  "${PYTHON[@]}" -m agentweaver.simulator.replay \
    --processed "$PROCESSED_DIR" \
    --wafer-config configs/wafer_4x4.yaml \
    --policy "$POLICY" \
    --run-id "$RUN_ID" \
    --out "$TMP_DIR/replay_${POLICY}.csv"
done

"${PYTHON[@]}" - <<'PY'
import csv
import glob
import os
from pathlib import Path

tmp = Path(os.environ["TMP_DIR"])
results = Path(os.environ["RESULTS_DIR"])
rows = []
fields = []
for path in sorted(glob.glob(str(tmp / "replay_*.csv"))):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            for key in row:
                if key not in fields:
                    fields.append(key)
summary = results / "wafer_replay_summary_h100calib_pr2_v2_1.csv"
with summary.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
ablation = [
    r for r in rows
    if r.get("scenario") == "AGGREGATE"
    and r.get("policy") in {
        "naive_wafer", "static_branch_pinning", "wafer_fcfs",
        "acd_only", "acd_bes", "acd_nisp", "full_agentweaver",
    }
]
with (results / "ablation_h100calib_pr2_v2_1.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(ablation)
print(summary)
PY

"${PYTHON[@]}" - <<'PY'
import csv
import os
import subprocess
from pathlib import Path

import yaml

tmp = Path(os.environ["TMP_DIR"])
processed = os.environ["PROCESSED_DIR"]
results = Path(os.environ["RESULTS_DIR"])
python_cmd = os.environ.get("PYTHON_CMD")
cmd_prefix = python_cmd.split() if python_cmd else ["uv", "run", "python"]

def read_aggregate(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))[-1]

def write(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

rows = []
for name, cfg in [("4x4", "configs/wafer_4x4.yaml"), ("6x6", "configs/wafer_6x6.yaml"), ("8x8", "configs/wafer_8x8.yaml")]:
    out = tmp / f"mesh_{name}.csv"
    subprocess.check_call(cmd_prefix + [
        "-m", "agentweaver.simulator.replay",
        "--processed", processed,
        "--wafer-config", cfg,
        "--policy", "full_agentweaver",
        "--run-id", os.environ["RUN_ID"],
        "--out", str(out),
    ])
    agg = read_aggregate(out)
    agg["mesh"] = name
    rows.append(agg)
write(results / "sensitivity_mesh_h100calib_pr2_v2_1.csv", rows)

base = yaml.safe_load(open("configs/wafer_4x4.yaml", encoding="utf-8"))
rows = []
for bw in [0.25, 0.5, 1, 2, 4, 8]:
    cfg = dict(base)
    cfg["link_bandwidth_TBps"] = bw
    cfg_path = tmp / f"wafer_bw_{bw}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp / f"bw_{bw}.csv"
    subprocess.check_call(cmd_prefix + [
        "-m", "agentweaver.simulator.replay",
        "--processed", processed,
        "--wafer-config", str(cfg_path),
        "--policy", "full_agentweaver",
        "--run-id", os.environ["RUN_ID"],
        "--out", str(out),
    ])
    agg = read_aggregate(out)
    agg["link_bandwidth_TBps"] = bw
    rows.append(agg)
write(results / "sensitivity_link_bw_h100calib_pr2_v2_1.csv", rows)

summary = list(csv.DictReader(open(results / "wafer_replay_summary_h100calib_pr2_v2_1.csv", newline="", encoding="utf-8")))
rows = []
for mult in [0, 0.5, 1, 2, 4]:
    for row in summary:
        if row["scenario"] == "S3_tool_stall_heavy" and row["policy"] in {"naive_wafer", "acd_nisp", "full_agentweaver"}:
            out = dict(row)
            out["tool_latency_multiplier"] = mult
            out["projected_jct"] = float(row["jct"]) + (mult - 1) * float(row["tool_blocked_region_time"])
            rows.append(out)
write(results / "sensitivity_tool_latency_h100calib_pr2_v2_1.csv", rows)
PY

echo "H100-calibrated synthetic replay complete"
ls -1 \
  "$RESULTS_DIR/wafer_replay_summary_h100calib_pr2_v2_1.csv" \
  "$RESULTS_DIR/ablation_h100calib_pr2_v2_1.csv" \
  "$RESULTS_DIR/sensitivity_mesh_h100calib_pr2_v2_1.csv" \
  "$RESULTS_DIR/sensitivity_link_bw_h100calib_pr2_v2_1.csv" \
  "$RESULTS_DIR/sensitivity_tool_latency_h100calib_pr2_v2_1.csv"
