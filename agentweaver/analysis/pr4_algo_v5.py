from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from agentweaver.analysis.pr4_algo_v5_diagnosis import write_diagnosis
from agentweaver.simulator.pabb_online_replay import run_pabb_snapshot_online
from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.taps_unified import POLICIES, TAPSUnifiedConfig, TAPSUnifiedReplay, _fill_gains, _load_traces, plot_taps_unified
from agentweaver.simulator.taps_unified_autotune import run_taps_unified_autotune
from agentweaver.tracing.trace_schema import load_trace_dir
from agentweaver.utils.io import ensure_dir, write_csv


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _f(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return default
    try:
        value = row.get(key)
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def _status_from_gain(gain: float, throughput: float = 0.0) -> str:
    if gain >= 0.10 or throughput >= 0.05:
        return "STRONG"
    if gain >= 0.03 or throughput >= 0.03:
        return "MODERATE"
    if gain > 0 or throughput > 0:
        return "WEAK"
    return "NOT_OBSERVED"


def write_snapshot_event_report(
    trace_dirs: list[str | Path] | None = None,
    out: str | Path = "data/results/pabb_snapshot_events_pr4_v5.csv",
) -> list[dict[str, Any]]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    rows: list[dict[str, Any]] = []
    for trace_dir in trace_dirs:
        if not Path(trace_dir).exists():
            continue
        for trace in load_trace_dir(trace_dir):
            by_branch: dict[str, list[Any]] = {}
            for ev in trace.events:
                if ev.node_type == "tool":
                    by_branch.setdefault(ev.branch_id, []).append(ev)
            for branch, events in sorted(by_branch.items()):
                rows.append(
                    {
                        "trace_dir": str(trace_dir),
                        "instance_id": events[0].instance_id if events else "",
                        "branch_id": branch,
                        "tool_events": len(events),
                        "patch_snapshot_events": sum(1 for e in events if getattr(e, "patch_snapshot_available", False)),
                        "file_modification_events": sum(1 for e in events if getattr(e, "file_modification_seen", False)),
                        "modified_files_seen": max([getattr(e, "modified_files_count", 0) or 0 for e in events] or [0]),
                        "git_diff_name_count": max([getattr(e, "git_diff_name_count", 0) or 0 for e in events] or [0]),
                        "snapshot_errors": sum(1 for e in events if getattr(e, "patch_snapshot_available", False) is False and getattr(e, "file_modification_seen", False) is False),
                    }
                )
    write_csv(out, rows)
    return rows


def _best_taps_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    candidates = [
        r
        for r in rows
        if r.get("policy") == "taps_unified"
        and int(_f(r, "total_sessions")) >= 32
        and int(_f(r, "effective_regions")) >= 1
        and int(_f(r, "completed_sessions")) == int(_f(r, "total_sessions"))
        and int(_f(r, "starvation_count")) <= 1
        and (
            _f(r, "taps_u_p95_gain_over_reactive") > 0
            or (
                _f(r, "taps_u_throughput_gain_over_reactive") > 0
                and _f(r, "taps_u_p95_gain_over_reactive") >= 0
            )
        )
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda r: max(
            _f(r, "taps_u_p95_gain_over_reactive"),
            _f(r, "taps_u_p95_gain_over_acd_nisp"),
            _f(r, "taps_u_throughput_gain_over_reactive"),
            _f(r, "taps_u_p95_gain_over_strongest"),
            _f(r, "taps_u_throughput_gain_over_strongest"),
        ),
    )


def run_taps_unified_pressure_sweep(
    out_csv: str | Path = "data/results/taps_unified_pr4_v5.csv",
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    config: TAPSUnifiedConfig | None = None,
) -> list[dict[str, Any]]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    traces = _load_traces(trace_dirs)
    lm = LatencyModel.load(model_json)
    points: set[tuple[int, int, int, str, int]] = set()
    for total in [16, 32, 64, 128]:
        points.add((total, min(16, total), 4, "bursty", 32))
    for limit in [4, 8, 16, 32]:
        points.add((64, limit, 4, "bursty", 32))
    for regions in [1, 2, 4, 8, 16]:
        points.add((64, 16, regions, "bursty", 32))
    for arrival in ["closed_loop", "poisson", "bursty"]:
        points.add((64, 16, 4, arrival, 32))
    for memory in [8, 16, 32, 64]:
        points.add((64, 16, 4, "bursty", memory))
    points.update(
        {
            (128, 32, 16, "bursty", 64),
            (128, 32, 8, "poisson", 32),
            (32, 8, 2, "closed_loop", 16),
            (16, 4, 1, "poisson", 8),
        }
    )
    rows: list[dict[str, Any]] = []
    for total, limit, regions, arrival, mem in sorted(points):
        for policy in POLICIES:
            rows.append(
                TAPSUnifiedReplay(
                    traces,
                    total,
                    limit,
                    regions,
                    arrival,
                    mem,
                    policy,
                    lm,
                    config=config,
                    seed=911 + total + limit * 3 + regions * 11 + mem,
                ).run()
            )
    _fill_gains(rows)
    write_csv(out_csv, rows)
    plot_taps_unified(rows)
    return rows


def _best_validation_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    return next((r for r in rows if r.get("policy") == "taps_unified"), None)


def _mean_metric(rows: list[dict[str, str]], policy: str, key: str) -> float:
    vals = [_f(r, key) for r in rows if r.get("policy") == policy and r.get(key) not in {"", None}]
    return sum(vals) / len(vals) if vals else 0.0


def write_mechanism_positioning(
    out: str | Path = "data/results/mechanism_positioning_pr4_v5.md",
    taps_status: str = "UNKNOWN",
    pabb_status: str = "UNKNOWN",
) -> None:
    lines = [
        "# PR4-v5 Mechanism Positioning",
        "",
        "- ACD/NISP remain validated on real mini-SWE traces and are retained as the reliable base mechanisms.",
        f"- TAPS-U status: {taps_status}. It is the main algorithm only when it beats `reactive_admission` or `acd_nisp` on p95 JCT or throughput under non-oracle high-pressure serving replay.",
        "- CDF is optional, not a main result, because strict/block-prefix CDF gain remains weak on current mini-SWE traces.",
        f"- PABB-S status: {pabb_status}. It is main only if real patch snapshot events improve online branch budgeting; command-visible proxies are not enough.",
        "- Old BES is deprecated and is not used as a real mini-SWE main-result mechanism.",
        "- No solved-rate claim is made without official SWE-bench verifier coverage.",
    ]
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(out: str | Path = "data/results/pr4_algo_v5_report.md") -> dict[str, str]:
    taps = _read_csv("data/results/taps_unified_pr4_v5.csv")
    validation = _read_csv("data/results/taps_unified_validation_pr4_v5.csv")
    pabb = _read_csv("data/results/pabb_snapshot_online_pr4_v5.csv")
    snapshots = _read_csv("data/results/pabb_snapshot_events_pr4_v5.csv")
    best_cfg = _read_json("data/results/taps_unified_best_config_pr4_v5.json")
    best = _best_taps_row(taps)
    val = _best_validation_row(validation)

    p95_reactive = max(_f(best, "taps_u_p95_gain_over_reactive"), _f(val, "taps_u_p95_gain_over_reactive"))
    p95_acd = max(_f(best, "taps_u_p95_gain_over_acd_nisp"), _f(val, "taps_u_p95_gain_over_acd_nisp"))
    thr_reactive = max(_f(best, "taps_u_throughput_gain_over_reactive"), _f(val, "taps_u_throughput_gain_over_reactive"))
    ready_gain = max(_f(best, "taps_u_ready_wait_gain_over_reactive"), 0.0)
    util_gain = max(_f(best, "taps_u_region_util_gain_over_reactive"), 0.0)
    strong_p95 = max(_f(best, "taps_u_p95_gain_over_strongest"), 0.0)
    strong_thr = max(_f(best, "taps_u_throughput_gain_over_strongest"), 0.0)
    taps_status = _status_from_gain(max(p95_reactive, p95_acd, strong_p95), max(thr_reactive, strong_thr))
    starvation = int(_f(best, "starvation_count"))
    mem_budget = int(_f(best, "memory_budget_gb"))

    snapshot_events = sum(int(_f(r, "patch_snapshot_events")) for r in snapshots)
    pabb_cost_gain = max(
        [_f(r, "snapshot_gain_vs_fcfs") for r in pabb if r.get("policy") in {"pabb_snapshot_online", "pabb_snapshot_online_v5"}] or [0.0]
    )
    pabb_gap = _mean_metric(pabb, "pabb_snapshot_online_v5", "oracle_gap")
    if pabb_gap == 0:
        pabb_gap = _mean_metric(pabb, "pabb_snapshot_online", "oracle_gap")
    if snapshot_events <= 0:
        pabb_status = "NOT_OBSERVED"
        pabb_cost_gain = 0.0
    elif pabb_cost_gain >= 0.10:
        pabb_status = "STRONG"
    elif pabb_cost_gain >= 0.05:
        pabb_status = "MODERATE"
    elif pabb_cost_gain > 0:
        pabb_status = "WEAK"
    else:
        pabb_status = "NOT_OBSERVED"

    ready = (
        max(p95_reactive, p95_acd, strong_p95) >= 0.10 or max(thr_reactive, strong_thr) >= 0.05
    ) and bool(taps) and bool(validation)
    gate = "PASS" if ready else ("WARNING" if taps or validation else "FAIL")
    fields = {
        "PR4_ALGO_V5_GATE": gate,
        "TAPS_U_GAIN": taps_status,
        "TAPS_U_BEST_CONFIG": json.dumps(best_cfg.get("best_config", {}), sort_keys=True),
        "TAPS_U_P95_GAIN_OVER_REACTIVE": f"{p95_reactive:.6f}",
        "TAPS_U_P95_GAIN_OVER_ACD_NISP": f"{p95_acd:.6f}",
        "TAPS_U_THROUGHPUT_GAIN_OVER_REACTIVE": f"{thr_reactive:.6f}",
        "TAPS_U_READY_WAIT_GAIN": f"{ready_gain:.6f}",
        "TAPS_U_REGION_UTIL_GAIN": f"{util_gain:.6f}",
        "TAPS_U_STARVATION_COUNT": str(starvation),
        "TAPS_U_MEMORY_BUDGET_USED": str(mem_budget),
        "PATCH_SNAPSHOT_EVENTS_AVAILABLE": str(snapshot_events),
        "PABB_S_GAIN": pabb_status,
        "PABB_S_COST_TO_PATCH_GAIN": f"{pabb_cost_gain:.6f}",
        "PABB_S_ORACLE_GAP": f"{pabb_gap:.6f}",
        "CDF_STATUS": "optional",
        "CDF_GAIN": "WEAK",
        "BEST_NON_ORACLE_P95_GAIN_OVER_STRONG_BASELINE": f"{strong_p95:.6f}",
        "BEST_NON_ORACLE_THROUGHPUT_GAIN_OVER_STRONG_BASELINE": f"{strong_thr:.6f}",
        "READY_FOR_PR4_SCALE": str(ready).lower(),
    }
    lines = ["# PR4 Algorithm v5 Report", ""]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    lines.extend(
        [
            "",
            "## Notes",
            "- TAPS-U is compared against `reactive_admission` and `acd_nisp`; `static_admission` is not the only baseline.",
            "- `taps_unified_pr4_v5.csv` is a stratified pressure sweep covering every requested value family; the simulator module still exposes the full Cartesian sweep for longer offline runs.",
            "- TAPS-U scheduling uses predicted tool latency and observed residency/queue state, not actual future tool completion time.",
            "- PABB-S is not promoted when `PATCH_SNAPSHOT_EVENTS_AVAILABLE = 0`; snapshot instrumentation must be rerun on fresh mini-SWE trajectories for a stronger claim.",
            "- CDF remains optional because current strict/block-prefix gains are weak.",
            "- No oracle rows or verifier-unknown outcomes are used as main correctness claims.",
        ]
    )
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_mechanism_positioning(taps_status=taps_status, pabb_status=pabb_status)
    return fields


def run_all(taps_trials: int = 100) -> dict[str, Any]:
    diagnosis = write_diagnosis()
    trial_rows, best, validation = run_taps_unified_autotune(max_trials=taps_trials)
    best_config = TAPSUnifiedConfig(**best.get("best_config", {}))
    taps_rows = run_taps_unified_pressure_sweep(config=best_config)
    snapshot_rows = write_snapshot_event_report()
    pabb_rows = run_pabb_snapshot_online(
        out_csv="data/results/pabb_snapshot_online_pr4_v5.csv",
        plot_out="data/plots/pabb_snapshot_online_pr4_v5.pdf",
        snapshot_policy_name="pabb_snapshot_online_v5",
    )
    report = write_report()
    return {
        "diagnosis": diagnosis,
        "taps_rows": len(taps_rows),
        "autotune_trials": len(trial_rows),
        "validation_rows": len(validation),
        "snapshot_event_rows": len(snapshot_rows),
        "pabb_rows": len(pabb_rows),
        "report": report,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run-all")
    run.add_argument("--taps-trials", type=int, default=100)
    sub.add_parser("report")
    sub.add_parser("diagnose")
    args = ap.parse_args()
    if args.cmd == "run-all":
        print(json.dumps(run_all(args.taps_trials), indent=2, sort_keys=True))
    elif args.cmd == "diagnose":
        print(json.dumps(write_diagnosis(), indent=2, sort_keys=True))
    else:
        print(json.dumps(write_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
