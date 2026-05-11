from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.analysis.context_segment_graph import process_trace_dir
from agentweaver.profiling.pr2_v2 import real_policy_comparison
from agentweaver.simulator.context_domain_factorization import compare_context_reuse
from agentweaver.simulator.multisession_replay import run_multisession
from agentweaver.simulator.progress_aware_branch_budgeting import run_pabb
from agentweaver.simulator.replay import replay
from agentweaver.utils.io import ensure_dir, write_csv


CDF_POLICIES = ["naive_wafer", "acd_only", "acd_cdf", "acd_nisp", "acd_cdf_nisp", "full_agentweaver_cdf"]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return default
    try:
        value = row.get(key, "")
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("instance_id") == "AGGREGATE"), rows[-1] if rows else {})


def run_cdf_policy_replay(
    trace_dir: str | Path = "data/traces/mini_swe_lite10_r4_timed",
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    wafer_config: str | Path = "configs/wafer_6x6.yaml",
    processed_dir: str | Path = "data/processed/mini_swe_lite10_r4_timed_pr4_cdf",
    out_csv: str | Path = "data/results/mini_swe_lite10_r4_timed_cdf_policy_comparison.csv",
) -> list[dict[str, Any]]:
    processed = ensure_dir(processed_dir)
    process_trace_dir(trace_dir, processed, "configs/default.yaml")
    shutil.copyfile(model_json, processed / "h100_latency_model.json")
    tmp = ensure_dir(Path("data/results/.tmp_pr4_cdf_replay"))
    rows: list[dict[str, Any]] = []
    fields: list[str] = []
    for policy in CDF_POLICIES:
        policy_rows = replay(processed, wafer_config, policy, tmp / f"{policy}.csv", run_id="pr4_algo_cdf")
        agg = dict(_aggregate(policy_rows))
        agg["status"] = "ok"
        rows.append(agg)
        for key in agg:
            if key not in fields:
                fields.append(key)
    write_csv(out_csv, rows, fields)
    shutil.rmtree(tmp, ignore_errors=True)
    return rows


def write_mechanism_positioning(out: str | Path = "data/results/mechanism_positioning_pr4_algo.md") -> None:
    text = """# PR4 Algorithm Mechanism Positioning

ACD/CDF:
Main context-domain mechanism. ACD exploits naturally occurring exact-prefix shared context in the trace, while CDF is a replay-level canonical prompt rendering potential that factors invariant task/repo/tool context into stable context domains. The target is repeated prefill and shared repo-history context.

NISP/TAPS:
Main tool-stall runtime mechanism. NISP handles per-branch state parking across tool stalls. TAPS handles multi-session stall hiding by releasing compute regions while tools run and scheduling ready LLM work using locality, resume urgency, criticality, and age.

PABB:
Branch-budget mechanism. PABB replaces old BES for real coding-agent workloads. It uses patch, test, tool, duplicate-patch, token-cost, and optional official verifier signals. When verifier results are unknown, PABB reports patch/progress metrics only, not solved rate.

Old BES:
Deprecated or folded into PABB. It is not used as the real mini-SWE main-result mechanism because PR3-v4 and PR3-v3 did not observe independent BES gain on timed mini-SWE traces.

Real mini-SWE main result:
Attribute real-trace gains to ACD/CDF/NISP/TAPS and report PABB only as branch-budget/progress control. Do not attribute real mini-SWE gains to old BES.
"""
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")


def _avg_metric(rows: list[dict[str, str]], policy: str, metric: str) -> float | None:
    vals = [_f(r, metric, float("nan")) for r in rows if r.get("budget_policy") == policy and r.get(metric) not in {"", None}]
    vals = [v for v in vals if v == v]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _pabb_budget_gains(rows: list[dict[str, str]]) -> tuple[float, float, bool]:
    fcfs = {
        (r.get("instance_id"), r.get("max_active_branches"), r.get("max_steps_per_branch")): r
        for r in rows
        if r.get("budget_policy") == "fcfs_budget"
    }
    time_gains: list[float] = []
    token_gains: list[float] = []
    coverage_gain = False
    for row in rows:
        if row.get("budget_policy") != "pabb_budget":
            continue
        key = (row.get("instance_id"), row.get("max_active_branches"), row.get("max_steps_per_branch"))
        base = fcfs.get(key)
        pabb_cost = _f(row, "cost_to_first_nonempty_patch", float("nan"))
        pabb_tokens = _f(row, "tokens_to_first_nonempty_patch", float("nan"))
        if pabb_cost != pabb_cost:
            continue
        fcfs_cost = _f(base, "cost_to_first_nonempty_patch", float("nan"))
        fcfs_tokens = _f(base, "tokens_to_first_nonempty_patch", float("nan"))
        if fcfs_cost != fcfs_cost:
            coverage_gain = True
            time_gains.append(1.0)
        elif fcfs_cost > 0:
            time_gains.append((fcfs_cost - pabb_cost) / fcfs_cost)
        if pabb_tokens != pabb_tokens:
            continue
        if fcfs_tokens != fcfs_tokens:
            coverage_gain = True
            token_gains.append(1.0)
        elif fcfs_tokens > 0:
            token_gains.append((fcfs_tokens - pabb_tokens) / fcfs_tokens)
    time_gain = sum(time_gains) / len(time_gains) if time_gains else 0.0
    token_gain = sum(token_gains) / len(token_gains) if token_gains else 0.0
    return time_gain, token_gain, coverage_gain


def plot_progression(
    cdf_rows: list[dict[str, Any]],
    taps_rows: list[dict[str, str]],
    pabb_rows: list[dict[str, str]],
    out: str | Path = "data/plots/agentweaver_progression_pr4_algo.pdf",
) -> None:
    by_policy = {r.get("policy"): r for r in cdf_rows}
    naive = _f(by_policy.get("naive_wafer"), "jct", 1.0) or 1.0
    stages = ["Naive", "+ACD", "+CDF", "+NISP", "+TAPS", "+PABB"]
    values = [
        1.0,
        _f(by_policy.get("acd_only"), "jct", naive) / naive,
        _f(by_policy.get("acd_cdf"), "jct", naive) / naive,
        _f(by_policy.get("acd_cdf_nisp"), "jct", naive) / naive,
    ]
    taps8 = {r.get("policy"): r for r in taps_rows if str(r.get("sessions")) == "8"}
    p95_naive = _f(taps8.get("naive_wafer"), "p95_jct", naive) or naive
    values.append(_f(taps8.get("taps"), "p95_jct", p95_naive) / p95_naive)
    pabb_gain, _, _ = _pabb_budget_gains(pabb_rows)
    values.append(max(0.0, 1.0 - pabb_gain) if pabb_gain else 1.0)
    ensure_dir(Path(out).parent)
    plt.figure(figsize=(7.6, 3.8))
    plt.plot(stages, values, marker="o")
    plt.axhline(1.0, color="black", linewidth=0.8)
    plt.ylabel("normalized replay cost (lower is better)")
    plt.xticks(rotation=20, ha="right")
    plt.title("Mechanism progression; TAPS/PABB use workload-specific metrics", fontsize=9)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def write_pr4_report(
    out: str | Path = "data/results/pr4_algo_report.md",
    cdf_csv: str | Path = "data/results/cdf_context_reuse_comparison.csv",
    taps_csv: str | Path = "data/results/multisession_taps_pr4_algo.csv",
    pabb_csv: str | Path = "data/results/pabb_branch_budget_pr4_algo.csv",
    positioning: str | Path = "data/results/mechanism_positioning_pr4_algo.md",
) -> dict[str, str]:
    cdf_rows = _read_csv(cdf_csv)
    taps_rows = _read_csv(taps_csv)
    pabb_rows = _read_csv(pabb_csv)
    repeated = sum(_f(r, "original_repeated_prefill_tokens") for r in cdf_rows)
    original = sum(_f(r, "original_exact_prefix_reusable_tokens") for r in cdf_rows)
    cdf_exact = sum(_f(r, "cdf_exact_prefix_reusable_tokens") for r in cdf_rows)
    cdf_added = sum(_f(r, "cdf_added_reusable_tokens") for r in cdf_rows)
    before_time = sum(_f(r, "estimated_prefill_time_before") for r in cdf_rows)
    saved_time = sum(_f(r, "estimated_prefill_time_saved") for r in cdf_rows)
    taps8 = {r.get("policy"): r for r in taps_rows if str(r.get("sessions")) == "8"}
    naive8 = taps8.get("naive_wafer")
    taps = taps8.get("taps")
    throughput_gain = _ratio(_f(taps, "throughput_sessions_per_sec") - _f(naive8, "throughput_sessions_per_sec"), _f(naive8, "throughput_sessions_per_sec"))
    p95_gain = _ratio(_f(naive8, "p95_jct") - _f(taps, "p95_jct"), _f(naive8, "p95_jct"))
    util_gain = _f(taps, "region_utilization") - _f(naive8, "region_utilization")
    time_to_patch_gain, tokens_to_patch_gain, pabb_coverage_gain = _pabb_budget_gains(pabb_rows)
    official = any("OFFICIAL_VERIFIER_USED = true" in Path(p).read_text(encoding="utf-8") for p in [Path("data/results/pr3_v4_report.md")] if p.exists())
    required = [
        Path(cdf_csv),
        Path(taps_csv),
        Path(pabb_csv),
        Path(positioning),
        Path("data/plots/cdf_context_reuse_pr4_algo.pdf"),
        Path("data/plots/multisession_taps_throughput_pr4_algo.pdf"),
        Path("data/plots/pabb_cost_to_patch_pr4_algo.pdf"),
        Path("data/plots/agentweaver_progression_pr4_algo.pdf"),
    ]
    ready = all(p.exists() for p in required)
    fields = {
        "PR4_ALGO_GATE": "PASS" if ready else "FAIL",
        "CDF_GAIN": "OBSERVED" if cdf_added > 0 else "NOT_OBSERVED",
        "CDF_ADDED_REUSABLE_TOKENS": str(int(cdf_added)),
        "CDF_REUSABLE_RATIO_BEFORE": f"{_ratio(original, repeated):.6f}",
        "CDF_REUSABLE_RATIO_AFTER": f"{_ratio(cdf_exact, repeated):.6f}",
        "CDF_MODEL_SIDE_SPEEDUP": f"{_ratio(saved_time, before_time):.6f}",
        "TAPS_GAIN": "OBSERVED" if throughput_gain > 0 or p95_gain > 0 or util_gain > 0 else "NOT_OBSERVED",
        "TAPS_THROUGHPUT_GAIN_AT_8_SESSIONS": f"{throughput_gain:.6f}",
        "TAPS_P95_JCT_GAIN_AT_8_SESSIONS": f"{p95_gain:.6f}",
        "TAPS_REGION_UTIL_GAIN_AT_8_SESSIONS": f"{util_gain:.6f}",
        "PABB_GAIN": "OBSERVED" if time_to_patch_gain > 0 or tokens_to_patch_gain > 0 or pabb_coverage_gain else "NOT_OBSERVED",
        "PABB_TIME_TO_PATCH_GAIN": f"{time_to_patch_gain:.6f}",
        "PABB_TOKENS_TO_PATCH_GAIN": f"{tokens_to_patch_gain:.6f}",
        "OFFICIAL_VERIFIER_USED": str(official).lower(),
        "BES_DEPRECATED": "true",
        "READY_FOR_PR4_SCALE": str(ready).lower(),
    }
    lines = ["# PR4 Algorithm Report", ""]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    lines.extend(
        [
            "",
            "## Notes",
            "- CDF results are canonicalized replay potential; raw traces are unchanged.",
            "- CDF_GAIN may be NOT_OBSERVED when the collected mini-SWE traces already expose almost all repeated shared context as exact-prefix reusable.",
            "- TAPS is a multi-session replay using measured mini-SWE tool latencies and H100-calibrated model latency.",
            "- TAPS can improve throughput/p95 while region_utilization drops if context reuse removes enough model-side work in a tool-dominated trace.",
            "- PABB reports patch/progress cost metrics; no solved rate is reported from unknown verifier results.",
            "- Old BES is deprecated for real mini-SWE main-result attribution.",
        ]
    )
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def run_all() -> dict[str, Any]:
    cdf_rows = compare_context_reuse()
    cdf_policy = run_cdf_policy_replay()
    taps_rows = run_multisession()
    pabb_rows = run_pabb()
    write_mechanism_positioning()
    plot_progression(cdf_policy, _read_csv("data/results/multisession_taps_pr4_algo.csv"), _read_csv("data/results/pabb_branch_budget_pr4_algo.csv"))
    report = write_pr4_report()
    return {
        "cdf_rows": len(cdf_rows),
        "cdf_policy_rows": len(cdf_policy),
        "taps_rows": len(taps_rows),
        "pabb_rows": len(pabb_rows),
        "report": report,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run-all")
    sub.add_parser("cdf-policy")
    sub.add_parser("positioning")
    sub.add_parser("progression-plot")
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "run-all":
        print(json.dumps(run_all(), indent=2))
    elif args.cmd == "cdf-policy":
        print(json.dumps({"rows": len(run_cdf_policy_replay())}, indent=2))
    elif args.cmd == "positioning":
        write_mechanism_positioning()
        print(json.dumps({"out": "data/results/mechanism_positioning_pr4_algo.md"}, indent=2))
    elif args.cmd == "progression-plot":
        plot_progression(
            _read_csv("data/results/mini_swe_lite10_r4_timed_cdf_policy_comparison.csv"),
            _read_csv("data/results/multisession_taps_pr4_algo.csv"),
            _read_csv("data/results/pabb_branch_budget_pr4_algo.csv"),
        )
        print(json.dumps({"out": "data/plots/agentweaver_progression_pr4_algo.pdf"}, indent=2))
    elif args.cmd == "report":
        print(json.dumps(write_pr4_report(), indent=2))


if __name__ == "__main__":
    main()
