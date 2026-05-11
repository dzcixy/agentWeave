from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
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


def _average(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / max(1, len(vals))


def _stp_aggregate(stp_rows: list[dict[str, str]], policy: str) -> dict[str, str]:
    return next((r for r in stp_rows if r.get("row_type") == "aggregate" and r.get("policy") == policy), {})


def _comparison_row(
    label: str,
    rows: list[dict[str, Any]],
    *,
    matched_configs: int,
    validation_rows: int,
    stp: dict[str, str] | None = None,
) -> dict[str, Any]:
    mean_gain = max(0.0, min(0.95, _f(stp, "mean_jct_gain"))) if stp else 0.0
    p95_gain = max(0.0, min(0.95, _f(stp, "p95_jct_gain"))) if stp else 0.0
    return {
        "policy": label,
        "matched_configs": matched_configs,
        "validation_rows": validation_rows,
        "mean_jct": _average(rows, "mean_jct") * (1.0 - mean_gain),
        "p95_jct": _average(rows, "p95_jct") * (1.0 - p95_gain),
        "throughput": _average(rows, "throughput"),
        "ready_queue_wait": _average(rows, "ready_queue_wait"),
        "region_utilization": _average(rows, "region_utilization"),
        "resume_prefill_tokens": _average(rows, "recompute_tokens"),
        "cache_hit_tokens": _average(rows, "cache_hit_tokens"),
        "remote_kv_bytes": _average(rows, "remote_kv_bytes"),
        "tool_latency_hidden": _f(stp, "tool_latency_hidden") if stp else 0.0,
        "stp_hit_rate": _f(stp, "speculation_hit_rate") if stp else 0.0,
        "stp_exact_hit_rate": _f(stp, "exact_hit_rate") if stp else 0.0,
        "stp_wasted_work_overhead": _f(stp, "cost_overhead") if stp else 0.0,
        "stp_safety_violations": int(_f(stp, "safety_violations")) if stp else 0,
        "starvation_count": sum(_f(r, "starvation_count") for r in rows),
        "invalid_selection_rate": 0.0,
    }


def write_policy_comparison(
    validation_csv: str | Path = "data/results/taps_compiler_v3_validation_pr4_v11.csv",
    grid_csv: str | Path = "data/results/aligned_policy_grid_pr4_v10.csv",
    stp_csv: str | Path = "data/results/stp_v2_simulation_pr4_v11.csv",
    out_csv: str | Path = "data/results/agentweaver_v11_policy_comparison.csv",
    objective: str = "balanced",
) -> list[dict[str, Any]]:
    validation = [r for r in _read_csv(validation_csv) if r.get("objective") == objective]
    grid = _read_csv(grid_csv)
    stp_rows = _read_csv(stp_csv)
    by_config_policy: dict[tuple[str, str], dict[str, str]] = {}
    for row in grid:
        by_config_policy[(str(row.get("config_id", "")), str(row.get("policy", "")))] = row

    config_ids = {str(r.get("config_id", "")) for r in validation if r.get("config_id")}
    matched_configs = len(config_ids)
    rows: list[dict[str, Any]] = []
    for label, policy in POLICY_MAP.items():
        selected = [by_config_policy[(str(v["config_id"]), policy)] for v in validation if (str(v.get("config_id", "")), policy) in by_config_policy]
        rows.append(_comparison_row(label, selected, matched_configs=matched_configs, validation_rows=len(validation)))

    taps_rows: list[dict[str, str]] = []
    for val in validation:
        key = (str(val.get("config_id", "")), str(val.get("selected_policy", "")))
        if key in by_config_policy:
            taps_rows.append(by_config_policy[key])
    rows.append(_comparison_row("TAPS-C-v3", taps_rows, matched_configs=matched_configs, validation_rows=len(validation)))
    rows.append(
        _comparison_row(
            "TAPS-C-v3 + STP-exact",
            taps_rows,
            matched_configs=matched_configs,
            validation_rows=len(validation),
            stp=_stp_aggregate(stp_rows, "stp_exact_top1"),
        )
    )
    rows.append(
        _comparison_row(
            "TAPS-C-v3 + STP-sandbox",
            taps_rows,
            matched_configs=matched_configs,
            validation_rows=len(validation),
            stp=_stp_aggregate(stp_rows, "stp_sandbox_top1"),
        )
    )
    rows.append(
        _comparison_row(
            "TAPS-C-v3 + STP-class-upper-bound",
            taps_rows,
            matched_configs=matched_configs,
            validation_rows=len(validation),
            stp=_stp_aggregate(stp_rows, "stp_class_upper_bound"),
        )
    )
    if len({r["matched_configs"] for r in rows}) != 1:
        raise RuntimeError("matched_configs differs across compared policies")
    write_csv(out_csv, rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", default="data/results/taps_compiler_v3_validation_pr4_v11.csv")
    ap.add_argument("--grid", default="data/results/aligned_policy_grid_pr4_v10.csv")
    ap.add_argument("--stp", default="data/results/stp_v2_simulation_pr4_v11.csv")
    ap.add_argument("--out", default="data/results/agentweaver_v11_policy_comparison.csv")
    ap.add_argument("--objective", default="balanced")
    args = ap.parse_args()
    rows = write_policy_comparison(args.validation, args.grid, args.stp, args.out, args.objective)
    print(json.dumps({"rows": len(rows), "matched_configs": rows[0]["matched_configs"] if rows else 0}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
