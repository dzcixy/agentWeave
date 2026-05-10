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
export RUN_ID OUT_DIR RESULTS PROCESSED TRACE_DIR
rm -rf "$TRACE_DIR" "$PROCESSED" "$TMP"
mkdir -p "$TRACE_DIR" "$PROCESSED" "$RESULTS" "$PLOTS" "$TMP"
trap 'rm -rf "$TMP"' EXIT

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
import csv
import os
from pathlib import Path

results = Path(os.environ["RESULTS"])
run_id = os.environ["RUN_ID"]

def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

replay = rows(results / "wafer_replay_summary.csv")
acd = rows(results / "acd_mapping.csv")
trace_summary = rows(results / "synthetic_trace_summary.csv")

def r(scenario, policy):
    for row in replay:
        if row.get("scenario") == scenario and row.get("policy") == policy:
            return row
    raise KeyError((scenario, policy))

def f(row, key):
    return float(row.get(key, 0) or 0)

def i(row, key):
    return int(float(row.get(key, 0) or 0))

policies = {row.get("policy") for row in replay}
required = {"naive_wafer","static_branch_pinning","wafer_fcfs","acd_only","acd_bes","acd_nisp","full_agentweaver"}
policy_ok = required.issubset(policies)
trace_ok = all(int(row["llm_events"]) == 2 * int(row["branch_fanout"]) and int(row["tool_events"]) == int(row["branch_fanout"]) for row in trace_summary)
acd_before = sum(float(x["avg_hops_before"]) for x in acd) / max(1, len(acd))
acd_after = sum(float(x["avg_hops_after"]) for x in acd) / max(1, len(acd))
noc_before = sum(float(x["noc_bytes_before"]) for x in acd)
noc_after = sum(float(x["noc_bytes_after"]) for x in acd)
hotspot_before = float(acd[0]["hotspot_before"]) if acd else 0
hotspot_after = float(acd[0]["hotspot_after"]) if acd else 0
s1_acd_ok = acd_after < acd_before or noc_after < noc_before
s1_hotspot_ok = hotspot_after <= hotspot_before or noc_after < noc_before

s2_naive = r("S2_branch_heavy", "naive_wafer")
s2_full = r("S2_branch_heavy", "full_agentweaver")
s2_bes = r("S2_branch_heavy", "acd_bes")
s2_ok = i(s2_full, "branch_wasted_tokens") < i(s2_naive, "branch_wasted_tokens") or i(s2_bes, "branch_wasted_tokens") < i(s2_naive, "branch_wasted_tokens")
s2_block_ok = f(s2_full, "blocked_compute_time_avoided") > 0 or f(s2_bes, "blocked_compute_time_avoided") > 0

s3_naive = r("S3_tool_stall_heavy", "naive_wafer")
s3_nisp = r("S3_tool_stall_heavy", "acd_nisp")
s3_full = r("S3_tool_stall_heavy", "full_agentweaver")
s3_resume_ok = i(s3_nisp, "resume_prefill_tokens") < i(s3_naive, "resume_prefill_tokens") or i(s3_full, "resume_prefill_tokens") < i(s3_naive, "resume_prefill_tokens")
parking_states = sum(1 for key in ["hot_count","warm_count","cold_count"] if i(s3_nisp, key) + i(s3_full, key) > 0)
s3_parking_ok = parking_states >= 2

def benefit(scenario):
    naive = r(scenario, "naive_wafer")
    full = r(scenario, "full_agentweaver")
    return (f(naive, "jct") - f(full, "jct")) / max(1e-9, f(naive, "jct"))

s1_benefit = benefit("S1_context_heavy")
s2_benefit = benefit("S2_branch_heavy")
s4_benefit = benefit("S4_low_reuse_negative")
s5_benefit = benefit("S5_tool_dominated_negative")
s4_ok = s4_benefit < max(s1_benefit, s2_benefit)
s5_ok = s5_benefit < max(s1_benefit, s2_benefit)

gate = all([policy_ok, trace_ok, s1_acd_ok, s1_hotspot_ok, s2_ok, s2_block_ok, s3_resume_ok, s3_parking_ok, s4_ok, s5_ok])
body = f"""# PR-1 Final Gate Report

## Validation

- `bash scripts/run_all_small.sh --run-id {run_id}`: completed.
- `pytest -q`: must be run by the caller before this script; PR2 phase 0 records the actual result separately.
- Required policies present: {policy_ok}
- Synthetic branch shape LLM0 -> TOOL -> LLM1 -> VERIFIER: {trace_ok}

## Mechanism Checks

### S1_context_heavy

- avg_hops_before = {acd_before:.6f}
- avg_hops_after = {acd_after:.6f}
- noc_bytes_before = {noc_before:.0f}
- noc_bytes_after = {noc_after:.0f}
- hotspot_before = {hotspot_before:.6f}
- hotspot_after = {hotspot_after:.6f}
- ACD locality check = {s1_acd_ok}
- hotspot/NoC check = {s1_hotspot_ok}

### S2_branch_heavy

- naive branch_wasted_tokens = {i(s2_naive, "branch_wasted_tokens")}
- full branch_wasted_tokens = {i(s2_full, "branch_wasted_tokens")}
- acd_bes branch_wasted_tokens = {i(s2_bes, "branch_wasted_tokens")}
- full blocked_compute_time_avoided = {f(s2_full, "blocked_compute_time_avoided"):.6f}
- acd_bes blocked_compute_time_avoided = {f(s2_bes, "blocked_compute_time_avoided"):.6f}
- branch waste check = {s2_ok}
- blocked compute release check = {s2_block_ok}

### S3_tool_stall_heavy

- naive resume_prefill_tokens = {i(s3_naive, "resume_prefill_tokens")}
- acd_nisp resume_prefill_tokens = {i(s3_nisp, "resume_prefill_tokens")}
- full resume_prefill_tokens = {i(s3_full, "resume_prefill_tokens")}
- acd_nisp HOT/WARM/COLD = {i(s3_nisp, "hot_count")}/{i(s3_nisp, "warm_count")}/{i(s3_nisp, "cold_count")}
- full HOT/WARM/COLD = {i(s3_full, "hot_count")}/{i(s3_full, "warm_count")}/{i(s3_full, "cold_count")}
- resume prefill check = {s3_resume_ok}
- parking diversity check = {s3_parking_ok}

### Negative Controls

- S1 benefit = {s1_benefit:.6f}
- S2 benefit = {s2_benefit:.6f}
- S4_low_reuse_negative benefit = {s4_benefit:.6f}
- S5_tool_dominated_negative benefit = {s5_benefit:.6f}
- S4 low-reuse reduced-benefit check = {s4_ok}
- S5 tool-dominated reduced-benefit check = {s5_ok}

## Remaining Limitations

- Default latency model is still placeholder until PR2 H100 profiling succeeds.
- Wafer results remain trace-driven simulation.

PR1_GATE = {"PASS" if gate else "FAIL"}
"""

for name in ["pr1_report.md", f"{run_id}_report.md"]:
    path = results / name
    path.write_text(body, encoding="utf-8")
    print(path)
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
      data/results/pr1_report.md \
      "$RESULTS/${RUN_ID}_report.md"
