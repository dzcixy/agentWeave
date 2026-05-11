from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from agentweaver.utils.io import write_csv


POLICY_MAP = {
    "reactive_admission": "reactive_admission",
    "acd_nisp": "acd_nisp",
    "taps_admission_v4": "taps_admission_v4",
    "taps_domain_v4_fixed": "taps_domain_v4",
    "taps_unified_v5_fixed": "taps_unified_v5",
}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return default
    try:
        value = row.get(key)
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / max(1, len(vals))


def _aggregate_stp(stp_rows: list[dict[str, str]], policy: str) -> dict[str, str]:
    return next((r for r in stp_rows if r.get("row_type") == "aggregate" and r.get("policy") == policy), {})


def _schedule_by_policy(schedule_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in schedule_rows:
        groups.setdefault(str(row.get("policy", "")), []).append(row)
    out: dict[str, dict[str, float]] = {}
    for policy, rows in groups.items():
        out[policy] = {
            "cache_hit_tokens": _avg(rows, "cached_tokens"),
            "resume_prefill_tokens": _avg(rows, "recompute_tokens"),
            "local_context_bytes": _avg(rows, "local_context_bytes"),
            "remote_context_bytes": _avg(rows, "remote_context_bytes"),
            "remote_kv_bytes": _avg(rows, "schedule_remote_kv_bytes"),
            "tool_latency_hidden": _avg(rows, "tool_latency_hidden"),
        }
    return out


def _schedule_metrics_for(label: str, schedule: dict[str, dict[str, float]]) -> dict[str, float]:
    if label == "acd_nisp":
        return schedule.get("acd_nisp", {})
    if label.startswith("TAPS-C-v3 + STP-AE") or label == "full AgentWeaver":
        return schedule.get("full AgentWeaver", schedule.get("TAPS-C-v3 + STP-AE-top3", schedule.get("TAPS-C-v3", {})))
    if label.startswith("TAPS-C-v3") or label.startswith("taps_"):
        return schedule.get("TAPS-C-v3", {})
    return {}


def _comparison_row(
    label: str,
    rows: list[dict[str, Any]],
    *,
    matched_configs: int,
    validation_rows: int,
    schedule: dict[str, dict[str, float]],
    stp: dict[str, str] | None = None,
) -> dict[str, Any]:
    mean_gain = max(0.0, min(0.95, _f(stp, "mean_jct_gain"))) if stp else 0.0
    p95_gain = max(0.0, min(0.95, _f(stp, "p95_jct_gain"))) if stp else 0.0
    sched = _schedule_metrics_for(label, schedule)
    stp_hidden = _f(stp, "tool_latency_hidden") if stp else sched.get("tool_latency_hidden", 0.0)
    return {
        "policy": label,
        "matched_configs": matched_configs,
        "validation_rows": validation_rows,
        "mean_jct": _avg(rows, "mean_jct") * (1.0 - mean_gain),
        "p95_jct": _avg(rows, "p95_jct") * (1.0 - p95_gain),
        "throughput": _avg(rows, "throughput"),
        "ready_queue_wait": _avg(rows, "ready_queue_wait"),
        "region_utilization": _avg(rows, "region_utilization"),
        "memory_occupancy": _avg(rows, "memory_occupancy"),
        "resume_prefill_tokens": sched.get("resume_prefill_tokens", _avg(rows, "recompute_tokens")),
        "cache_hit_tokens": sched.get("cache_hit_tokens", _avg(rows, "cache_hit_tokens")),
        "local_context_bytes": sched.get("local_context_bytes", 0.0),
        "remote_context_bytes": sched.get("remote_context_bytes", 0.0),
        "remote_kv_bytes": sched.get("remote_kv_bytes", _avg(rows, "remote_kv_bytes")),
        "tool_latency_hidden": stp_hidden,
        "artifact_hit_rate": _f(stp, "artifact_hit_rate") if stp else 0.0,
        "exact_command_hit_rate": _f(stp, "exact_command_hit_rate") if stp else 0.0,
        "stp_wasted_work_overhead": _f(stp, "cost_overhead") if stp else 0.0,
        "stp_safety_violations": int(_f(stp, "safety_violations")) if stp else 0,
        "starvation_count": sum(_f(r, "starvation_count") for r in rows),
        "invalid_selection_rate": 0.0,
    }


def write_policy_comparison(
    validation_csv: str | Path = "data/results/taps_compiler_v3_validation_pr4_v11.csv",
    grid_csv: str | Path = "data/results/aligned_policy_grid_pr4_v10.csv",
    stp_ae_csv: str | Path = "data/results/stp_ae_simulation_pr4_v12.csv",
    schedule_csv: str | Path = "data/results/schedule_summary_pr4_v12.csv",
    out_csv: str | Path = "data/results/agentweaver_v12_policy_comparison.csv",
    objective: str = "balanced",
) -> list[dict[str, Any]]:
    validation = [r for r in _read_csv(validation_csv) if r.get("objective") == objective]
    grid = _read_csv(grid_csv)
    stp_rows = _read_csv(stp_ae_csv)
    schedule = _schedule_by_policy(_read_csv(schedule_csv))
    by_config_policy = {(str(r.get("config_id", "")), str(r.get("policy", ""))): r for r in grid}
    config_ids = {str(r.get("config_id", "")) for r in validation if r.get("config_id")}
    matched_configs = len(config_ids)
    rows: list[dict[str, Any]] = []
    for label, policy in POLICY_MAP.items():
        sub = [by_config_policy[(str(v["config_id"]), policy)] for v in validation if (str(v.get("config_id", "")), policy) in by_config_policy]
        rows.append(_comparison_row(label, sub, matched_configs=matched_configs, validation_rows=len(validation), schedule=schedule))
    taps_rows: list[dict[str, str]] = []
    for val in validation:
        key = (str(val.get("config_id", "")), str(val.get("selected_policy", "")))
        if key in by_config_policy:
            taps_rows.append(by_config_policy[key])
    rows.append(_comparison_row("TAPS-C-v3", taps_rows, matched_configs=matched_configs, validation_rows=len(validation), schedule=schedule))
    rows.append(_comparison_row("TAPS-C-v3 + STP-exact", taps_rows, matched_configs=matched_configs, validation_rows=len(validation), schedule=schedule, stp=_aggregate_stp(stp_rows, "stp_exact_v2")))
    rows.append(_comparison_row("TAPS-C-v3 + STP-AE-top1", taps_rows, matched_configs=matched_configs, validation_rows=len(validation), schedule=schedule, stp=_aggregate_stp(stp_rows, "stp_ae_top1")))
    rows.append(_comparison_row("TAPS-C-v3 + STP-AE-top3", taps_rows, matched_configs=matched_configs, validation_rows=len(validation), schedule=schedule, stp=_aggregate_stp(stp_rows, "stp_ae_top3_budgeted")))
    rows.append(_comparison_row("TAPS-C-v3 + STP-AE-sandbox", taps_rows, matched_configs=matched_configs, validation_rows=len(validation), schedule=schedule, stp=_aggregate_stp(stp_rows, "stp_ae_sandbox")))
    rows.append(_comparison_row("STP-class-upper-bound", taps_rows, matched_configs=matched_configs, validation_rows=len(validation), schedule=schedule, stp=_aggregate_stp(stp_rows, "stp_class_upper_bound")))
    rows.append(_comparison_row("full AgentWeaver", taps_rows, matched_configs=matched_configs, validation_rows=len(validation), schedule=schedule, stp=_aggregate_stp(stp_rows, "stp_ae_top3_budgeted")))
    if len({r["matched_configs"] for r in rows}) != 1:
        raise RuntimeError("matched_configs differs across policies")
    write_csv(out_csv, rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", default="data/results/taps_compiler_v3_validation_pr4_v11.csv")
    ap.add_argument("--grid", default="data/results/aligned_policy_grid_pr4_v10.csv")
    ap.add_argument("--stp-ae", default="data/results/stp_ae_simulation_pr4_v12.csv")
    ap.add_argument("--schedule", default="data/results/schedule_summary_pr4_v12.csv")
    ap.add_argument("--out", default="data/results/agentweaver_v12_policy_comparison.csv")
    args = ap.parse_args()
    rows = write_policy_comparison(args.validation, args.grid, args.stp_ae, args.schedule, args.out)
    print(json.dumps({"rows": len(rows), "matched_configs": rows[0]["matched_configs"] if rows else 0}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
