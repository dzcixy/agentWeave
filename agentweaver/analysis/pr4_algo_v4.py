from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from agentweaver.analysis.pr4_algo_v4_diagnosis import write_diagnosis
from agentweaver.simulator.algorithm_optimizer import run_optimizer
from agentweaver.simulator.pabb_online_replay import run_pabb_snapshot_online
from agentweaver.simulator.taps_admission_control import run_admission
from agentweaver.simulator.taps_domain_scheduler import run_domain_scheduler
from agentweaver.simulator.taps_memory_budget import run_memory_budget
from agentweaver.utils.io import ensure_dir


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return default
    try:
        v = row.get(key, "")
        return default if v in ("", None) else float(v)
    except Exception:
        return default


def _status(gain: float) -> str:
    if gain >= 0.10:
        return "STRONG"
    if gain >= 0.03:
        return "MODERATE"
    if gain > 0:
        return "WEAK"
    return "NOT_OBSERVED"


def _max_metric(rows: list[dict[str, str]], metric: str, policy: str | None = None) -> float:
    vals = [_f(r, metric) for r in rows if policy is None or r.get("policy") == policy]
    return max(vals) if vals else 0.0


def _mean_metric(rows: list[dict[str, str]], metric: str, policy: str | None = None) -> float:
    vals = [_f(r, metric) for r in rows if (policy is None or r.get("policy") == policy) and r.get(metric) not in {"", None}]
    return sum(vals) / len(vals) if vals else 0.0


def _run_pabb_tests() -> str:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        ["python", "-m", "pytest", "-q", "tests/test_pabb_no_future_leakage.py"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    p = Path("data/results/pabb_no_future_leakage_tests_pr4_v4.txt")
    ensure_dir(p.parent)
    p.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return "PASS" if proc.returncode == 0 else "FAIL"


def write_report(out: str | Path = "data/results/pr4_algo_v4_report.md", pabb_test_status: str = "UNKNOWN") -> dict[str, str]:
    if pabb_test_status == "UNKNOWN":
        test_log = Path("data/results/pabb_no_future_leakage_tests_pr4_v4.txt")
        if test_log.exists() and "passed" in test_log.read_text(encoding="utf-8", errors="replace"):
            pabb_test_status = "PASS"
    domain = _read_csv("data/results/taps_domain_scheduler_pr4_v4.csv")
    admission = _read_csv("data/results/taps_admission_pr4_v4.csv")
    memory = _read_csv("data/results/taps_memory_budget_pr4_v4.csv")
    pabb = _read_csv("data/results/pabb_snapshot_online_pr4_v4.csv")
    validation = _read_csv("data/results/pr4_v4_validation_results.csv")
    cdf = _read_csv("data/results/cdf_strict_prefix_comparison_pr4_v2.csv")
    cdf_added = sum(_f(r, "cdf_added_reusable_tokens") for r in cdf)

    domain_remote = _max_metric(domain, "domain_remote_kv_reduction", "taps_domain")
    domain_p95 = _max_metric(domain, "domain_p95_jct_gain", "taps_domain")
    domain_mean = _max_metric(domain, "domain_mean_jct_gain", "taps_domain")
    domain_status = _status(max(domain_p95, domain_mean, domain_remote))

    adm_thr = _max_metric(admission, "admission_throughput_gain", "taps_admission")
    adm_p95 = _max_metric(admission, "admission_p95_jct_gain", "taps_admission")
    adm_util = _max_metric(admission, "admission_region_util_gain", "taps_admission")
    adm_status = _status(max(adm_thr, adm_p95))

    stable = [r for r in memory if str(r.get("stable_gain", "")).lower() in {"true", "1"}]
    min_budget = min([int(float(r["memory_budget_gb"])) for r in stable], default=0)
    mem_status = "PASS" if memory and min_budget else ("WARNING" if memory else "FAIL")

    snap_fcfs = _mean_metric(pabb, "snapshot_gain_vs_fcfs", "pabb_snapshot_online")
    snap_v3 = _mean_metric(pabb, "snapshot_gain_vs_pabb_v3", "pabb_snapshot_online")
    snap_file = 0.0
    fcfs_by_key = {
        (r.get("instance_id"), r.get("max_active_branches"), r.get("max_steps_per_branch")): r
        for r in pabb
        if r.get("policy") == "fcfs_budget"
    }
    file_gains: list[float] = []
    for row in pabb:
        if row.get("policy") != "pabb_snapshot_online":
            continue
        base = fcfs_by_key.get((row.get("instance_id"), row.get("max_active_branches"), row.get("max_steps_per_branch")))
        b = _f(base, "time_to_first_file_modification", float("nan"))
        s = _f(row, "time_to_first_file_modification", float("nan"))
        if b == b and s == s and b > 0:
            file_gains.append((b - s) / b)
    snap_file = sum(file_gains) / len(file_gains) if file_gains else 0.0
    pabb_gap = _mean_metric(pabb, "oracle_gap", "pabb_snapshot_online")
    snapshot_events = sum(int(_f(r, "snapshot_events_available")) for r in pabb)
    snap_status = _status(max(snap_fcfs, snap_v3, snap_file))
    if snapshot_events == 0 and snap_status in {"STRONG", "MODERATE"}:
        snap_status = "WEAK"

    best_validation_p95 = max([_f(r, "p95_gain_vs_base") for r in validation], default=0.0)
    best_validation_thr = max([_f(r, "throughput_gain_vs_base") for r in validation], default=0.0)
    best_non_oracle_p95 = max(domain_p95, adm_p95, best_validation_p95)
    best_non_oracle_thr = max(adm_thr, best_validation_thr)
    best_ready = _max_metric(domain, "domain_ready_wait_gain", "taps_domain")
    best_patch = max(snap_fcfs, snap_v3)

    required = [
        Path("data/results/pr4_algo_v4_diagnosis.md"),
        Path("data/results/taps_domain_scheduler_pr4_v4.csv"),
        Path("data/results/taps_admission_pr4_v4.csv"),
        Path("data/results/taps_memory_budget_pr4_v4.csv"),
        Path("data/results/pabb_snapshot_online_pr4_v4.csv"),
        Path("data/results/pr4_v4_optimizer_trials.csv"),
        Path("data/results/pr4_v4_best_configs.json"),
        Path("data/results/pr4_v4_validation_results.csv"),
    ]
    ready = (
        all(p.exists() for p in required)
        and pabb_test_status == "PASS"
        and mem_status in {"PASS", "WARNING"}
        and max(best_non_oracle_p95, best_non_oracle_thr) >= 0.10
    )
    fields = {
        "PR4_ALGO_V4_GATE": "PASS" if ready else "WARNING",
        "CDF_STATUS": "optional",
        "CDF_GAIN": "WEAK" if cdf_added > 0 else "NOT_OBSERVED",
        "DOMAIN_SCHEDULER_GAIN": domain_status,
        "DOMAIN_REMOTE_KV_REDUCTION": f"{domain_remote:.6f}",
        "DOMAIN_P95_JCT_GAIN": f"{domain_p95:.6f}",
        "ADMISSION_CONTROL_GAIN": adm_status,
        "ADMISSION_THROUGHPUT_GAIN": f"{adm_thr:.6f}",
        "ADMISSION_P95_JCT_GAIN": f"{adm_p95:.6f}",
        "ADMISSION_REGION_UTIL_GAIN": f"{adm_util:.6f}",
        "MEMORY_BUDGET_RESULTS": mem_status,
        "MIN_BUDGET_WITH_STABLE_GAIN": str(min_budget),
        "SNAPSHOT_PROGRESS_GAIN": snap_status,
        "PATCH_SNAPSHOT_EVENTS_AVAILABLE": str(snapshot_events),
        "TIME_TO_FILE_MODIFICATION_GAIN": f"{snap_file:.6f}",
        "COST_TO_PATCH_GAIN": f"{max(snap_fcfs, snap_v3):.6f}",
        "PABB_ORACLE_GAP": f"{pabb_gap:.6f}",
        "BEST_NON_ORACLE_P95_GAIN": f"{best_non_oracle_p95:.6f}",
        "BEST_NON_ORACLE_THROUGHPUT_GAIN": f"{best_non_oracle_thr:.6f}",
        "BEST_NON_ORACLE_READY_WAIT_GAIN": f"{best_ready:.6f}",
        "BEST_NON_ORACLE_PATCH_COST_GAIN": f"{best_patch:.6f}",
        "PABB_S_NO_FUTURE_LEAKAGE_TESTS": pabb_test_status,
        "OLD_BES_DEPRECATED": "true",
        "READY_FOR_PR4_SCALE": str(ready).lower(),
    }
    lines = ["# PR4 Algorithm v4 Report", ""]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Non-oracle mechanisms only are used for main status fields. Oracle rows are retained only as upper bounds in CSVs.",
            "- CDF is explicitly optional because current strict/block-prefix gain remains weak.",
            "- TAPS-D targets domain locality and remote KV traffic; TAPS-A targets serving admission under tool stalls; TAPS-M tests explicit memory pressure.",
            "- PABB-S uses only event-visible tool snapshot/file-modification signals. Existing PR3 traces may lack true patch snapshot fields; those cases are reported by snapshot_events_available in the CSV.",
            "- When PATCH_SNAPSHOT_EVENTS_AVAILABLE is 0, PABB-S gains are capped at WEAK because they come from command-visible file-modification proxies rather than real incremental git snapshots.",
            "- No solved-rate claim is made without official verifier coverage.",
        ]
    )
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def run_all(taps_trials: int = 50, pabb_trials: int = 20) -> dict[str, Any]:
    diagnosis = write_diagnosis()
    domain = run_domain_scheduler()
    admission = run_admission()
    memory = run_memory_budget()
    pabb = run_pabb_snapshot_online()
    trials, best, validation = run_optimizer(taps_trials=taps_trials, pabb_trials=pabb_trials)
    test_status = _run_pabb_tests()
    report = write_report(pabb_test_status=test_status)
    return {
        "diagnosis": diagnosis,
        "domain_rows": len(domain),
        "admission_rows": len(admission),
        "memory_rows": len(memory),
        "pabb_rows": len(pabb),
        "optimizer_trials": len(trials),
        "validation_rows": len(validation),
        "pabb_tests": test_status,
        "report": report,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run-all")
    run.add_argument("--taps-trials", type=int, default=50)
    run.add_argument("--pabb-trials", type=int, default=20)
    sub.add_parser("diagnosis")
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "run-all":
        print(json.dumps(run_all(args.taps_trials, args.pabb_trials), indent=2, sort_keys=True))
    elif args.cmd == "diagnosis":
        print(json.dumps(write_diagnosis(), indent=2, sort_keys=True))
    elif args.cmd == "report":
        print(json.dumps(write_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
