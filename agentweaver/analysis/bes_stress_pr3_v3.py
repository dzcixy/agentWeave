from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.analysis.context_segment_graph import process_trace_dir
from agentweaver.simulator.replay import replay
from agentweaver.utils.io import ensure_dir, read_csv, write_csv


POLICIES = ["acd_only", "acd_bes", "acd_nisp", "full_agentweaver"]


def _write_config(path: Path, mesh_rows: int, mesh_cols: int, effective_regions: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "wafer:",
                f"  mesh_rows: {mesh_rows}",
                f"  mesh_cols: {mesh_cols}",
                "  die_memory_capacity_gb: 80",
                "  kv_budget_ratio: 0.3",
                "  link_bandwidth_TBps: 1.0",
                "  link_latency_ns: 500",
                "  routing: xy",
                "  region_granularity: die",
                "  enable_replication: true",
                "  enable_noc_slack: true",
                f"  effective_regions: {effective_regions}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row.get("instance_id") == "AGGREGATE":
            return row
    return rows[-1] if rows else {}


def _f(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def run_bes_stress(
    trace_dir: str | Path = "data/traces/mini_swe_lite10_r4_timed",
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    out_csv: str | Path = "data/results/bes_stress_mini_swe_pr3_v3.csv",
    plot_out: str | Path = "data/plots/bes_stress_mini_swe_pr3_v3.pdf",
) -> list[dict[str, Any]]:
    processed = ensure_dir("data/processed/bes_stress_mini_swe_pr3_v3")
    process_trace_dir(trace_dir, processed, "configs/default.yaml")
    model_path = Path(model_json)
    if not model_path.exists():
        raise FileNotFoundError(f"missing latency model: {model_path}")
    shutil.copyfile(model_path, processed / "h100_latency_model.json")
    tmp = ensure_dir(Path("data/results/.tmp_bes_stress_pr3_v3"))
    rows: list[dict[str, Any]] = []
    for mesh_rows, mesh_cols in [(2, 2), (4, 4), (6, 6)]:
        max_regions = mesh_rows * mesh_cols
        for effective_regions in [1, 2, 4, 8]:
            if effective_regions > max_regions:
                continue
            config = processed / "configs" / f"wafer_{mesh_rows}x{mesh_cols}_r{effective_regions}.yaml"
            _write_config(config, mesh_rows, mesh_cols, effective_regions)
            by_policy: dict[str, dict[str, Any]] = {}
            for policy in POLICIES:
                policy_out = tmp / f"{mesh_rows}x{mesh_cols}_r{effective_regions}_{policy}.csv"
                policy_rows = replay(processed, config, policy, policy_out, run_id="bes_stress_pr3_v3")
                aggregate = _aggregate(policy_rows)
                by_policy[policy] = aggregate
                rows.append(
                    {
                        "run_id": "bes_stress_pr3_v3",
                        "mesh": f"{mesh_rows}x{mesh_cols}",
                        "mesh_rows": mesh_rows,
                        "mesh_cols": mesh_cols,
                        "effective_regions": effective_regions,
                        "branch_fanout": aggregate.get("branch_fanout", ""),
                        "policy": policy,
                        "jct": aggregate.get("jct", ""),
                        "branch_wait_time": aggregate.get("branch_wait_time", ""),
                        "region_utilization": aggregate.get("region_utilization", ""),
                        "blocked_compute_time_avoided": aggregate.get("blocked_compute_time_avoided", ""),
                        "time_to_first_success": aggregate.get("time_to_first_success", ""),
                        "branch_wasted_tokens": aggregate.get("branch_wasted_tokens", ""),
                        "acd_bes_gain_vs_acd_only": "",
                        "full_gain_vs_acd_nisp": "",
                    }
                )
            acd = _f(by_policy.get("acd_only", {}), "jct")
            bes = _f(by_policy.get("acd_bes", {}), "jct")
            nisp = _f(by_policy.get("acd_nisp", {}), "jct")
            full = _f(by_policy.get("full_agentweaver", {}), "jct")
            bes_gain = (acd - bes) / acd if acd > 0 else math.nan
            full_gain = (nisp - full) / nisp if nisp > 0 else math.nan
            for row in rows:
                if row["mesh_rows"] == mesh_rows and row["mesh_cols"] == mesh_cols and row["effective_regions"] == effective_regions:
                    row["acd_bes_gain_vs_acd_only"] = "" if math.isnan(bes_gain) else bes_gain
                    row["full_gain_vs_acd_nisp"] = "" if math.isnan(full_gain) else full_gain
    write_csv(out_csv, rows)
    _plot(rows, plot_out)
    shutil.rmtree(tmp, ignore_errors=True)
    return rows


def _plot(rows: list[dict[str, Any]], out: str | Path) -> None:
    ensure_dir(Path(out).parent)
    agg = [r for r in rows if r.get("policy") in POLICIES]
    labels = []
    by_policy = {policy: [] for policy in POLICIES}
    for mesh, eff in sorted({(r["mesh"], int(r["effective_regions"])) for r in agg}, key=lambda x: (x[0], x[1])):
        labels.append(f"{mesh}\nr{eff}")
        for policy in POLICIES:
            row = next((r for r in agg if r["mesh"] == mesh and int(r["effective_regions"]) == eff and r["policy"] == policy), None)
            by_policy[policy].append(_f(row or {}, "jct"))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    xs = range(len(labels))
    for policy, ys in by_policy.items():
        axes[0].plot(xs, ys, marker="o", label=policy)
    axes[0].set_xticks(list(xs), labels, rotation=45, ha="right")
    axes[0].set_ylabel("aggregate JCT (s)")
    axes[0].legend(fontsize=8)
    gain_rows = [r for r in agg if r["policy"] == "acd_bes"]
    axes[1].bar(range(len(gain_rows)), [float(r["acd_bes_gain_vs_acd_only"] or 0.0) for r in gain_rows])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(range(len(gain_rows)), [f"{r['mesh']}\nr{r['effective_regions']}" for r in gain_rows], rotation=45, ha="right")
    axes[1].set_ylabel("BES gain vs ACD-only")
    fig.suptitle("BES stress on timed mini-SWE traces")
    fig.savefig(out)
    plt.close(fig)


def summarize_effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    configs = {(r["mesh"], r["effective_regions"]) for r in rows}
    gains = []
    for row in rows:
        if row.get("policy") == "acd_bes" and row.get("acd_bes_gain_vs_acd_only") not in {"", None}:
            gains.append(float(row["acd_bes_gain_vs_acd_only"]))
    max_gain = max(gains) if gains else 0.0
    return {
        "BES_STRESS_EVALUATION": "PASS" if len(configs) >= 3 else "FAIL",
        "BES_REAL_TRACE_EFFECT": "OBSERVED" if max_gain > 1e-6 else "NOT_OBSERVED",
        "num_resource_configurations": len(configs),
        "max_acd_bes_gain_vs_acd_only": max_gain,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default="data/traces/mini_swe_lite10_r4_timed")
    ap.add_argument("--model-json", default="data/profiles/h100_latency_model_pr2_v2.json")
    ap.add_argument("--out", default="data/results/bes_stress_mini_swe_pr3_v3.csv")
    ap.add_argument("--plot-out", default="data/plots/bes_stress_mini_swe_pr3_v3.pdf")
    args = ap.parse_args()
    rows = run_bes_stress(args.trace_dir, args.model_json, args.out, args.plot_out)
    print(json.dumps(summarize_effect(rows), indent=2))


if __name__ == "__main__":
    main()
