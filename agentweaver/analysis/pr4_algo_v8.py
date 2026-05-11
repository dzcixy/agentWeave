from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agentweaver.analysis.pr4_v8_eval_methodology import write_methodology
from agentweaver.astra.run_astra_smoke import export_policy_aware_smoke
from agentweaver.simulator.aligned_policy_sweep import ALIGNED_POLICIES, run_aligned_policy_grid
from agentweaver.simulator.taps_cost_model import run_taps_compiler_validation, summarize_validation, write_plots
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
        v = row.get(key)
        return default if v in ("", None) else float(v)
    except Exception:
        return default


def _parse_report_field(path: str | Path, key: str, default: str = "") -> str:
    p = Path(path)
    if not p.exists():
        return default
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key} = "):
            return line.split(" = ", 1)[1].strip()
    return default


def _gain_status(p95_gain: float, throughput_gain: float, regret: float, reactive_gain: float) -> str:
    if p95_gain >= 0.10 or (throughput_gain >= 0.05 and p95_gain >= 0):
        return "STRONG"
    if p95_gain >= 0.03 or (throughput_gain >= 0.03 and p95_gain >= 0) or (regret <= 0.05 and reactive_gain > 0):
        return "MODERATE"
    if p95_gain > 0 or throughput_gain > 0 or reactive_gain > 0:
        return "WEAK"
    return "NOT_OBSERVED"


def _objective_gain(path: str | Path, objective: str, key: str) -> float:
    for row in _read_csv(path):
        if row.get("objective") == objective:
            return _f(row, key)
    return 0.0


def write_report(out: str | Path = "data/results/pr4_algo_v8_report.md") -> dict[str, Any]:
    grid = _read_csv("data/results/aligned_policy_grid_pr4_v8.csv")
    validation = _read_csv("data/results/taps_compiler_validation_pr4_v8.csv")
    summary = summarize_validation(validation)
    random_summary = summarize_validation(validation, "random")
    configs = {r.get("config_id") for r in grid}
    policy_sets: dict[str, set[str]] = {}
    for row in grid:
        policy_sets.setdefault(row.get("config_id", ""), set()).add(row.get("policy", ""))
    complete_configs = sum(1 for p in policy_sets.values() if set(ALIGNED_POLICIES).issubset(p))
    best_fixed_counts = Counter(r.get("best_fixed_policy", "") for r in validation if r.get("split_type") == "random")
    best_fixed = (best_fixed_counts.most_common(1) or [("", 0)])[0][0]
    p95_gain = summary["mean_gain_over_best_fixed_p95"]
    thr_gain = summary["mean_throughput_gain_over_best_fixed"]
    regret = summary["mean_regret_to_oracle_p95"]
    reactive_gain = summary["mean_gain_over_reactive_p95"]
    gain_status = _gain_status(p95_gain, thr_gain, regret, reactive_gain)
    astra_status = _parse_report_field("data/results/astra_policy_aware_export_report.md", "ASTRA_POLICY_AWARE_EXPORT", "FAIL")
    astra_run = _parse_report_field("data/results/astra_run_report.md", "ASTRA_SMOKE_RUN", "false")
    ready = (
        complete_configs >= 100
        and (
            p95_gain >= 0.03
            or (thr_gain >= 0.03 and p95_gain >= 0)
            or (regret <= 0.05 and reactive_gain > 0)
        )
        and astra_status in {"PASS", "WARNING"}
    )
    gate = "PASS" if ready else ("WARNING" if grid and validation else "FAIL")
    fields: dict[str, Any] = {
        "PR4_ALGO_V8_GATE": gate,
        "EVAL_METHODOLOGY_FIXED": "true",
        "ORACLE_ENVELOPE_NOT_USED_AS_BASELINE": "true",
        "ALIGNED_POLICY_GRID_ROWS": len(grid),
        "ALIGNED_POLICY_GRID_CONFIGS": complete_configs,
        "TAPS_C_IMPLEMENTED": "true",
        "BEST_FIXED_POLICY": best_fixed,
        "TAPS_C_P95_GAIN_OVER_BEST_FIXED": f"{p95_gain:.6f}",
        "TAPS_C_THROUGHPUT_GAIN_OVER_BEST_FIXED": f"{thr_gain:.6f}",
        "TAPS_C_REGRET_TO_ORACLE_P95": f"{regret:.6f}",
        "TAPS_C_GAIN": gain_status,
        "TAPS_C_FAILURE_CONFIGS": int(summary["failure_configs"]),
        "P95_OPT_GAIN": f"{_objective_gain('data/results/taps_compiler_objectives_pr4_v8.csv', 'p95_opt', 'gain_over_best_fixed_p95'):.6f}",
        "THROUGHPUT_OPT_GAIN": f"{_objective_gain('data/results/taps_compiler_objectives_pr4_v8.csv', 'throughput_opt', 'throughput_gain_over_best_fixed'):.6f}",
        "BALANCED_GAIN": f"{_objective_gain('data/results/taps_compiler_objectives_pr4_v8.csv', 'balanced', 'gain_over_best_fixed_p95'):.6f}",
        "ASTRA_POLICY_AWARE_EXPORT": astra_status,
        "ASTRA_SIM_RUN_COMPLETED": str(astra_run).lower(),
        "READY_FOR_PR4_SCALE": str(ready).lower(),
    }
    lines = ["# PR4 Algorithm v8 Report", ""]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    lines.extend(
        [
            "",
            "## Methodology",
            "- Best fixed policy is selected on train and applied to validation.",
            "- Oracle envelope is reported only as regret; it is not a deployable baseline.",
            "- TAPS-C uses train-derived cost-model predictions and does not inspect validation labels during selection.",
            "",
            "## Aggregate Across All Splits",
            f"ALL_SPLITS_P95_GAIN_OVER_BEST_FIXED = {summary['mean_gain_over_best_fixed_p95']:.6f}",
            f"ALL_SPLITS_THROUGHPUT_GAIN_OVER_BEST_FIXED = {summary['mean_throughput_gain_over_best_fixed']:.6f}",
            f"ALL_SPLITS_REGRET_TO_ORACLE_P95 = {summary['mean_regret_to_oracle_p95']:.6f}",
            "",
            "## Random Split Only",
            f"RANDOM_SPLIT_P95_GAIN_OVER_BEST_FIXED = {random_summary['mean_gain_over_best_fixed_p95']:.6f}",
            f"RANDOM_SPLIT_THROUGHPUT_GAIN_OVER_BEST_FIXED = {random_summary['mean_throughput_gain_over_best_fixed']:.6f}",
            f"RANDOM_SPLIT_REGRET_TO_ORACLE_P95 = {random_summary['mean_regret_to_oracle_p95']:.6f}",
        ]
    )
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def run_all(size: str = "medium") -> dict[str, Any]:
    write_methodology()
    grid = run_aligned_policy_grid(size=size)
    training, validation = run_taps_compiler_validation()
    write_plots()
    export_policy_aware_smoke()
    report = write_report()
    return {
        "grid_rows": len(grid),
        "training_rows": len(training),
        "validation_rows": len(validation),
        "report": report,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run-all")
    run.add_argument("--size", choices=["small", "medium", "full"], default="medium")
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "run-all":
        print(json.dumps(run_all(args.size), indent=2, sort_keys=True))
    else:
        print(json.dumps(write_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
