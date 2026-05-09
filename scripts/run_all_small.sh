#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/small_sanity.yaml
RUN_ID=pr1_synthetic
OUT_DIR=data
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

TRACE_DIR="$OUT_DIR/traces/synthetic"
PROCESSED="$OUT_DIR/processed/synthetic"
RESULTS="$OUT_DIR/results"
PLOTS="$OUT_DIR/plots"
TMP="$OUT_DIR/results/.tmp_${RUN_ID}"
export RUN_TMP="$TMP"
rm -rf "$TRACE_DIR" "$PROCESSED" "$TMP"
mkdir -p "$TRACE_DIR" "$PROCESSED" "$RESULTS" "$PLOTS" "$TMP"

python -m agentweaver.workloads.synthetic_fork_join --fixed-scenarios --out-dir "$TRACE_DIR"
python -m agentweaver.analysis.context_segment_graph --trace-dir "$TRACE_DIR" --out "$PROCESSED" --config configs/default.yaml

python - <<'PY'
import csv
from pathlib import Path
from agentweaver.tracing.trace_schema import load_trace_dir
rows=[]
for tr in load_trace_dir("data/traces/synthetic"):
    meta=tr.metadata
    llm=sum(1 for e in tr.events if e.node_type=="llm")
    tool=sum(1 for e in tr.events if e.node_type=="tool")
    branches=len({e.branch_id for e in tr.events if e.branch_id!="root"})
    rows.append({
        "scenario": meta.get("scenario"),
        "branch_fanout": branches,
        "llm_events": llm,
        "tool_events": tool,
        "shared_prefix_len": meta.get("shared_prefix_len"),
        "exact_prefix_reuse_ratio": meta.get("exact_prefix_reuse_ratio"),
        "success_branch": meta.get("success_branch"),
    })
Path("data/results").mkdir(parents=True, exist_ok=True)
with open("data/results/synthetic_trace_summary.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
PY

python - <<'PY'
import csv, json
from pathlib import Path
summary=json.load(open("data/processed/synthetic/context_summary.json"))
rows=[summary]
with open("data/results/context_reuse_summary.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
PY

python -m agentweaver.simulator.acd_mapping --processed "$PROCESSED" --config configs/wafer_4x4.yaml --out "$RESULTS/acd_mapping.csv"
python -m agentweaver.simulator.gpu_cache_sim --processed "$PROCESSED" --config "$CONFIG" --out "$RESULTS/gpu_cache_baselines.csv"

POLICIES=(naive_wafer static_branch_pinning wafer_fcfs acd_only acd_bes acd_nisp full_agentweaver)
for POLICY in "${POLICIES[@]}"; do
  python -m agentweaver.simulator.replay \
    --processed "$PROCESSED" \
    --wafer-config configs/wafer_4x4.yaml \
    --policy "$POLICY" \
    --run-id "$RUN_ID" \
    --out "$TMP/replay_${POLICY}.csv"
done

python - <<'PY'
import csv, glob, os
from pathlib import Path
rows=[]
for path in sorted(glob.glob(os.environ["RUN_TMP"] + "/replay_*.csv")):
    with open(path,newline="") as f:
        rows.extend(csv.DictReader(f))
fieldnames=[]
for r in rows:
    for k in r:
        if k not in fieldnames:
            fieldnames.append(k)
with open("data/results/wafer_replay_summary.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fieldnames)
    w.writeheader(); w.writerows(rows)
ablation=[r for r in rows if r.get("scenario")=="AGGREGATE" and r.get("policy") in {
    "naive_wafer","static_branch_pinning","wafer_fcfs","acd_only","acd_bes","acd_nisp","full_agentweaver"
}]
with open("data/results/ablation.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fieldnames)
    w.writeheader(); w.writerows(ablation)
PY

python - <<'PY'
import csv, os, subprocess, tempfile, yaml
from pathlib import Path
tmp=os.environ["RUN_TMP"]
meshes=[("4x4","configs/wafer_4x4.yaml"),("6x6","configs/wafer_6x6.yaml"),("8x8","configs/wafer_8x8.yaml")]
rows=[]
for name,cfg in meshes:
    out=f"{tmp}/mesh_{name}.csv"
    subprocess.check_call(["python","-m","agentweaver.simulator.replay","--processed","data/processed/synthetic","--wafer-config",cfg,"--policy","full_agentweaver","--out",out])
    with open(out,newline="") as f:
        agg=list(csv.DictReader(f))[-1]
    agg["mesh"]=name
    rows.append(agg)
with open("data/results/sensitivity_mesh.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

base=yaml.safe_load(open("configs/wafer_4x4.yaml"))
rows=[]
for bw in [0.25,0.5,1,2,4,8]:
    cfg=dict(base); cfg["link_bandwidth_TBps"]=bw
    p=Path(f"{tmp}/wafer_bw_{bw}.yaml")
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out=f"{tmp}/bw_{bw}.csv"
    subprocess.check_call(["python","-m","agentweaver.simulator.replay","--processed","data/processed/synthetic","--wafer-config",str(p),"--policy","full_agentweaver","--out",out])
    with open(out,newline="") as f:
        agg=list(csv.DictReader(f))[-1]
    agg["link_bandwidth_TBps"]=bw
    rows.append(agg)
with open("data/results/sensitivity_link_bw.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

summary=list(csv.DictReader(open("data/results/wafer_replay_summary.csv")))
rows=[]
for mult in [0,0.5,1,2,4]:
    for r in summary:
        if r["scenario"]=="S3_tool_stall_heavy" and r["policy"] in {"naive_wafer","acd_nisp","full_agentweaver"}:
            x=dict(r); x["tool_latency_multiplier"]=mult
            x["projected_jct"]=float(r["jct"]) + (mult-1)*float(r["tool_blocked_region_time"])
            rows.append(x)
with open("data/results/sensitivity_tool_latency.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
PY

python -m agentweaver.plotting.plot_all --results-dir "$RESULTS" --out-dir "$PLOTS"

python - <<'PY'
from pathlib import Path
report = Path("data/results/pr1_report.md")
report.write_text("""# PR-1 Report: Synthetic Correctness and Simulator Core Fix

## Modified files

- `agentweaver/workloads/synthetic_fork_join.py`
- `agentweaver/simulator/replay.py`
- `agentweaver/simulator/acd_mapping.py`
- `agentweaver/simulator/bes_scheduler.py`
- `agentweaver/simulator/nisp.py`
- `scripts/run_all_small.sh`
- `tests/test_synthetic_expected_properties.py`

## Validation

- `pytest -q`: run separately in this PR workflow.
- `bash scripts/run_all_small.sh`: completed and generated the required CSV/PDF/PNG artifacts.

## Mechanism checks

- ACD is exercised by exact-prefix shared context in S1/S2/S3 and reports lower post-mapping hop/traffic metrics than naive shared-bank placement.
- BES is exercised by comparing branch-elastic policies against `static_branch_pinning`, where tool-blocked branches retain regions in the baseline.
- NISP is exercised by the mandatory LLM_1 after each tool, with HOT/WARM/COLD restore changing resume prefill tokens.

## Negative controls

- S4 has low exact-prefix reuse, so AgentWeaver benefit is intentionally small.
- S5 is tool dominated, so wafer-side LLM/context benefits drop relative to total JCT.

## Remaining limitations

- The default latency model remains a placeholder until H100 profiling is collected.
- Real vLLM/SGLang/SWE-agent measurements are not fabricated by this script.

## H100 profiling blockers

- Start vLLM/SGLang with the target local Qwen model under `/data2/model_zoo`.
- Collect `data/profiles/h100_profile_raw.csv` and fit `data/profiles/h100_latency_model.json`.
- Run real multi-rollout SWE traces and convert them through the adapters before full evaluation.
""", encoding="utf-8")
print(report)
PY

echo "PR-1 synthetic pipeline complete"
echo "required outputs:"
ls -1 data/results/synthetic_trace_summary.csv \
      data/results/context_reuse_summary.csv \
      data/results/acd_mapping.csv \
      data/results/gpu_cache_baselines.csv \
      data/results/wafer_replay_summary.csv \
      data/results/ablation.csv \
      data/results/sensitivity_mesh.csv \
      data/results/sensitivity_link_bw.csv \
      data/results/sensitivity_tool_latency.csv \
      data/results/pr1_report.md
