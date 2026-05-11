from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentweaver.utils.io import ensure_dir, write_csv


BASELINE_POLICIES = {
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


def _gain(base: float, selected: float, lower_better: bool = True) -> float:
    if base <= 0 or not math.isfinite(base) or not math.isfinite(selected):
        return 0.0
    return (base - selected) / base if lower_better else (selected - base) / base


def _avg(vals: list[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / max(1, len(vals))


def _std(vals: list[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) <= 1:
        return 0.0
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def _ci95(vals: list[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return 0.0
    return 1.96 * _std(vals) / math.sqrt(len(vals))


def _policy_row(groups: dict[str, dict[str, dict[str, str]]], config_id: str, policy: str) -> dict[str, str] | None:
    return groups.get(config_id, {}).get(policy)


def _selected_row(
    groups: dict[str, dict[str, dict[str, str]]],
    validation_row: dict[str, str],
) -> dict[str, str] | None:
    return _policy_row(groups, str(validation_row.get("config_id", "")), str(validation_row.get("selected_policy", "")))


def _stp_adjustments(stp_csv: str | Path | None) -> dict[str, dict[str, float]]:
    if not stp_csv:
        return {}
    rows = _read_csv(stp_csv)
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("row_type") != "aggregate":
            continue
        policy = str(row.get("policy", ""))
        if not policy or policy == "no_stp":
            continue
        out[policy] = {
            "mean_gain": max(0.0, min(0.95, _f(row, "mean_jct_gain"))),
            "p95_gain": max(0.0, min(0.95, _f(row, "p95_jct_gain"))),
        }
    return out


def _comparison_rows_for_selection(
    validation_row: dict[str, str],
    selected_grid_row: dict[str, str],
    baseline_rows: list[tuple[str, dict[str, str]]],
    *,
    stp_enabled: bool = False,
    stp_policy: str = "",
    stp_mean_gain: float = 0.0,
    stp_p95_gain: float = 0.0,
) -> list[dict[str, Any]]:
    selected_p95 = _f(selected_grid_row, "p95_jct") * (1.0 - stp_p95_gain)
    selected_mean = _f(selected_grid_row, "mean_jct") * (1.0 - stp_mean_gain)
    selected_thr = _f(selected_grid_row, "throughput")
    selected_wait = _f(selected_grid_row, "ready_queue_wait")
    selected_remote = _f(selected_grid_row, "remote_kv_bytes")
    out: list[dict[str, Any]] = []
    for baseline_label, baseline in baseline_rows:
        base_p95 = _f(baseline, "p95_jct")
        base_mean = _f(baseline, "mean_jct")
        base_thr = _f(baseline, "throughput")
        base_wait = _f(baseline, "ready_queue_wait")
        base_remote = _f(baseline, "remote_kv_bytes")
        out.append(
            {
                "split_type": validation_row.get("split_type", ""),
                "objective": validation_row.get("objective", ""),
                "config_id": validation_row.get("config_id", ""),
                "selected_policy": validation_row.get("selected_policy", ""),
                "baseline_policy": baseline_label,
                "selected_p95": selected_p95,
                "baseline_p95": base_p95,
                "p95_gain": _gain(base_p95, selected_p95),
                "selected_mean": selected_mean,
                "baseline_mean": base_mean,
                "mean_gain": _gain(base_mean, selected_mean),
                "selected_throughput": selected_thr,
                "baseline_throughput": base_thr,
                "throughput_gain": _gain(base_thr, selected_thr, lower_better=False),
                "selected_ready_wait": selected_wait,
                "baseline_ready_wait": base_wait,
                "ready_wait_gain": _gain(base_wait, selected_wait),
                "selected_remote_kv_bytes": selected_remote,
                "baseline_remote_kv_bytes": base_remote,
                "remote_kv_reduction": _gain(base_remote, selected_remote),
                "selected_cache_hit_tokens": _f(selected_grid_row, "cache_hit_tokens"),
                "selected_resume_prefill_tokens": _f(selected_grid_row, "recompute_tokens"),
                "stp_enabled": str(stp_enabled).lower(),
                "stp_policy": stp_policy,
                "stp_mean_gain": stp_mean_gain,
                "stp_p95_gain": stp_p95_gain,
            }
        )
    return out


def write_matched_comparison(
    validation_csv: str | Path = "data/results/taps_compiler_v3_validation_pr4_v11.csv",
    grid_csv: str | Path = "data/results/aligned_policy_grid_pr4_v10.csv",
    stp_csv: str | Path | None = "data/results/stp_v2_simulation_pr4_v11.csv",
    out_csv: str | Path = "data/results/matched_policy_comparison_pr4_v11.csv",
    out_md: str | Path = "data/results/matched_policy_comparison_summary_pr4_v11.md",
    objective_filter: str | None = "balanced",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validation = _read_csv(validation_csv)
    if objective_filter:
        validation = [r for r in validation if r.get("objective") == objective_filter]
    grid = _read_csv(grid_csv)
    groups: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in grid:
        groups[str(row.get("config_id", ""))][str(row.get("policy", ""))] = row

    adjustments = _stp_adjustments(stp_csv)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for val in validation:
        cid = str(val.get("config_id", ""))
        selected = _selected_row(groups, val)
        if not selected:
            missing.append(f"{cid}:selected:{val.get('selected_policy', '')}")
            continue
        baseline_rows: list[tuple[str, dict[str, str]]] = []
        seen: set[tuple[str, str]] = set()
        for label, policy in BASELINE_POLICIES.items():
            row = _policy_row(groups, cid, policy)
            if row:
                baseline_rows.append((label, row))
                seen.add((label, policy))
            else:
                missing.append(f"{cid}:baseline:{policy}")
        best_fixed = str(val.get("best_fixed_policy", ""))
        if best_fixed:
            row = _policy_row(groups, cid, best_fixed)
            if row:
                baseline_rows.append(("best_fixed", row))
            else:
                missing.append(f"{cid}:best_fixed:{best_fixed}")
        oracle = str(val.get("p95_oracle_policy") or val.get("oracle_policy") or "")
        if oracle:
            row = _policy_row(groups, cid, oracle)
            if row:
                baseline_rows.append(("oracle_p95", row))
            else:
                missing.append(f"{cid}:oracle:{oracle}")
        rows.extend(_comparison_rows_for_selection(val, selected, baseline_rows))
        for stp_policy in ("stp_exact_top1", "stp_sandbox_top1", "stp_class_upper_bound"):
            adj = adjustments.get(stp_policy)
            if not adj:
                continue
            rows.extend(
                _comparison_rows_for_selection(
                    val,
                    selected,
                    baseline_rows,
                    stp_enabled=True,
                    stp_policy=stp_policy,
                    stp_mean_gain=adj["mean_gain"],
                    stp_p95_gain=adj["p95_gain"],
                )
            )

    write_csv(out_csv, rows)
    summary = summarize_matched(rows, validation, missing)
    write_summary_md(summary, out_md)
    return rows, summary


def summarize_matched(rows: list[dict[str, Any]], validation_rows: list[dict[str, str]], missing: list[str] | None = None) -> dict[str, Any]:
    missing = missing or []
    base_rows = [r for r in rows if str(r.get("stp_enabled", "false")).lower() != "true"]

    def gains_for(baseline: str) -> list[float]:
        return [_f(r, "p95_gain") for r in base_rows if r.get("baseline_policy") == baseline]

    def thr_gains_for(baseline: str) -> list[float]:
        return [_f(r, "throughput_gain") for r in base_rows if r.get("baseline_policy") == baseline]

    def remote_for(baseline: str) -> list[float]:
        return [_f(r, "remote_kv_reduction") for r in base_rows if r.get("baseline_policy") == baseline]

    stp_rows = [r for r in rows if str(r.get("stp_enabled", "false")).lower() == "true"]
    stp_exact_best = [r for r in stp_rows if r.get("baseline_policy") == "best_fixed" and r.get("stp_policy") == "stp_exact_top1"]
    stp_sandbox_best = [r for r in stp_rows if r.get("baseline_policy") == "best_fixed" and r.get("stp_policy") == "stp_sandbox_top1"]
    best_fixed_gains = gains_for("best_fixed")
    best_fixed_thr = thr_gains_for("best_fixed")
    configs = {str(r.get("config_id", "")) for r in validation_rows if r.get("config_id")}
    complete = bool(validation_rows) and not missing and len(best_fixed_gains) == len(validation_rows)
    summary: dict[str, Any] = {
        "MATCHED_COMPARISON": "PASS" if complete else "FAIL",
        "validation_rows": len(validation_rows),
        "matched_rows": len(rows),
        "matched_configs": len(configs),
        "missing_matches": len(missing),
        "missing_examples": ";".join(missing[:20]),
        "gain_over_reactive": _avg(gains_for("reactive_admission")),
        "gain_over_acd_nisp": _avg(gains_for("acd_nisp")),
        "gain_over_best_fixed": _avg(best_fixed_gains),
        "regret_to_oracle": -_avg(gains_for("oracle_p95")),
        "p95_std_over_best_fixed": _std(best_fixed_gains),
        "p95_ci95_over_best_fixed": _ci95(best_fixed_gains),
        "throughput_gain_over_best_fixed": _avg(best_fixed_thr),
        "throughput_std_over_best_fixed": _std(best_fixed_thr),
        "throughput_ci95_over_best_fixed": _ci95(best_fixed_thr),
        "remote_kv_reduction_over_best_fixed": _avg(remote_for("best_fixed")),
        "stp_exact_p95_gain_over_best_fixed": _avg([_f(r, "p95_gain") for r in stp_exact_best]),
        "stp_sandbox_p95_gain_over_best_fixed": _avg([_f(r, "p95_gain") for r in stp_sandbox_best]),
        "number_of_configs": len(configs),
        "number_of_validation_rows": len(validation_rows),
    }
    return summary


def write_summary_md(summary: dict[str, Any], out_md: str | Path) -> None:
    lines = ["# Matched Policy Comparison PR4-v11", ""]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"{key} = {value:.6f}")
        else:
            lines.append(f"{key} = {value}")
    p = Path(out_md)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", default="data/results/taps_compiler_v3_validation_pr4_v11.csv")
    ap.add_argument("--grid", default="data/results/aligned_policy_grid_pr4_v10.csv")
    ap.add_argument("--stp", default="data/results/stp_v2_simulation_pr4_v11.csv")
    ap.add_argument("--out", default="data/results/matched_policy_comparison_pr4_v11.csv")
    ap.add_argument("--summary-out", default="data/results/matched_policy_comparison_summary_pr4_v11.md")
    ap.add_argument("--objective", default="balanced")
    args = ap.parse_args()
    rows, summary = write_matched_comparison(args.validation, args.grid, args.stp, args.out, args.summary_out, args.objective or None)
    print(json.dumps({"rows": len(rows), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
