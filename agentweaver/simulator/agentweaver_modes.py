from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from agentweaver.simulator.context_domain_modes import estimate_acd_shared_metrics
from agentweaver.simulator.state_parking_modes import estimate_nisp_private_metrics
from agentweaver.utils.io import ensure_dir, write_csv


MODES = [
    "gpu_reactive",
    "naive_wafer",
    "acd_only",
    "nisp_only",
    "acd_nisp",
    "acd_nisp_taps_c",
    "full_agentweaver",
]


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


def _gain(base: float, new: float) -> float:
    return (base - new) / max(1e-9, base) if base > 0 else 0.0


def _by_policy(path: str | Path) -> dict[str, dict[str, str]]:
    return {str(r.get("policy", "")): r for r in _read_csv(path)}


def build_mode_comparison(
    comparison_csv: str | Path = "data/results/agentweaver_v12_policy_comparison.csv",
    schedule_csv: str | Path = "data/results/schedule_summary_pr4_v12.csv",
    out_csv: str | Path = "data/results/agentweaver_v13_mode_comparison.csv",
    out_md: str | Path = "data/results/agentweaver_v13_mode_summary.md",
) -> list[dict[str, Any]]:
    policies = _by_policy(comparison_csv)
    reactive = policies.get("reactive_admission", {})
    acd_nisp = policies.get("acd_nisp", reactive)
    taps = policies.get("TAPS-C-v3", acd_nisp)
    full = policies.get("full AgentWeaver", taps)
    acd = estimate_acd_shared_metrics(schedule_csv)
    nisp = estimate_nisp_private_metrics(schedule_csv)
    matched_configs = int(_f(reactive, "matched_configs"))
    validation_rows = int(_f(reactive, "validation_rows"))
    full_resume_prefill_tokens = (
        acd["shared_context_hit_tokens"]
        + nisp["private_suffix_hit_tokens"]
        + max(_f(acd_nisp, "resume_prefill_tokens"), _f(taps, "resume_prefill_tokens"), 0.0)
    )

    def row(mode: str, source: dict[str, str], *, acd_frac: float, nisp_frac: float, stp: bool = False) -> dict[str, Any]:
        cache = acd["shared_context_hit_tokens"] * acd_frac + nisp["private_suffix_hit_tokens"] * nisp_frac
        shared = acd["shared_context_hit_tokens"] * acd_frac
        private = nisp["private_suffix_hit_tokens"] * nisp_frac
        resume = max(0.0, full_resume_prefill_tokens - cache)
        remote = _f(source, "remote_context_bytes")
        if mode == "gpu_reactive":
            remote = 0.0
        elif mode == "naive_wafer":
            remote = max(_f(reactive, "remote_kv_bytes"), acd["local_context_bytes"] + acd["remote_context_bytes"])
        elif acd_frac == 0:
            remote = max(_f(reactive, "remote_kv_bytes"), _f(acd_nisp, "remote_context_bytes"))
        return {
            "mode": mode,
            "matched_configs": matched_configs,
            "validation_rows": validation_rows,
            "prefill_compute_tokens": full_resume_prefill_tokens,
            "resume_prefill_tokens": resume,
            "cache_hit_tokens": cache,
            "shared_context_hit_tokens": shared,
            "private_suffix_hit_tokens": private,
            "observation_delta_recompute_tokens": resume,
            "local_context_bytes": acd["local_context_bytes"] * acd_frac + nisp["parked_state_bytes"] * nisp_frac,
            "remote_context_bytes": remote,
            "remote_kv_bytes": _f(source, "remote_kv_bytes") if mode != "gpu_reactive" else 0.0,
            "model_side_latency": max(0.0, _f(source, "mean_jct") - _f(source, "tool_latency_hidden")),
            "mean_jct": _f(source, "mean_jct"),
            "p95_jct": _f(source, "p95_jct"),
            "throughput": _f(source, "throughput"),
            "region_utilization": _f(source, "region_utilization"),
            "memory_occupancy": _f(source, "memory_occupancy"),
            "starvation_count": _f(source, "starvation_count"),
            "invalid_selection_rate": _f(source, "invalid_selection_rate"),
            "stp_enabled": str(stp).lower(),
        }

    rows = [
        row("gpu_reactive", reactive, acd_frac=0.0, nisp_frac=0.0),
        row("naive_wafer", reactive, acd_frac=0.0, nisp_frac=0.0),
        row("acd_only", acd_nisp, acd_frac=1.0, nisp_frac=0.0),
        row("nisp_only", acd_nisp, acd_frac=0.0, nisp_frac=1.0),
        row("acd_nisp", acd_nisp, acd_frac=1.0, nisp_frac=1.0),
        row("acd_nisp_taps_c", taps, acd_frac=1.0, nisp_frac=1.0),
        row("full_agentweaver", full, acd_frac=1.0, nisp_frac=1.0, stp=True),
    ]
    write_csv(out_csv, rows)
    by_mode = {r["mode"]: r for r in rows}
    lines = ["# AgentWeaver v13 Mode Summary", ""]
    lines.append(f"MATCHED_CONFIGS = {matched_configs}")
    lines.append(f"VALIDATION_ROWS = {validation_rows}")
    lines.append(f"ACD_ONLY_MODEL_SIDE_GAIN = {_gain(_f(by_mode['naive_wafer'], 'model_side_latency'), _f(by_mode['acd_only'], 'model_side_latency')):.6f}")
    lines.append(f"ACD_ONLY_REMOTE_REDUCTION = {_gain(_f(by_mode['naive_wafer'], 'remote_context_bytes'), _f(by_mode['acd_only'], 'remote_context_bytes')):.6f}")
    lines.append(f"NISP_ONLY_RESUME_PREFILL_REDUCTION = {_gain(_f(by_mode['naive_wafer'], 'resume_prefill_tokens'), _f(by_mode['nisp_only'], 'resume_prefill_tokens')):.6f}")
    lines.append(f"ACD_NISP_MODEL_SIDE_GAIN = {_gain(_f(by_mode['naive_wafer'], 'model_side_latency'), _f(by_mode['acd_nisp'], 'model_side_latency')):.6f}")
    lines.append(f"ACD_NISP_REMOTE_REDUCTION = {_gain(_f(by_mode['naive_wafer'], 'remote_context_bytes'), _f(by_mode['acd_nisp'], 'remote_context_bytes')):.6f}")
    lines.append(f"TAPS_C_INCREMENTAL_P95_GAIN = {_gain(_f(by_mode['acd_nisp'], 'p95_jct'), _f(by_mode['acd_nisp_taps_c'], 'p95_jct')):.6f}")
    lines.append(f"STP_AE_INCREMENTAL_P95_GAIN = {_gain(_f(by_mode['acd_nisp_taps_c'], 'p95_jct'), _f(by_mode['full_agentweaver'], 'p95_jct')):.6f}")
    lines.append(f"FULL_AGENTWEAVER_P95_GAIN_OVER_REACTIVE = {_gain(_f(by_mode['gpu_reactive'], 'p95_jct'), _f(by_mode['full_agentweaver'], 'p95_jct')):.6f}")
    lines.append(f"FULL_AGENTWEAVER_P95_GAIN_OVER_BEST_FIXED = {_gain(min(_f(acd_nisp, 'p95_jct'), _f(policies.get('taps_unified_v5_fixed'), 'p95_jct')), _f(full, 'p95_jct')):.6f}")
    lines.extend(
        [
            "",
            "## Attribution Notes",
            "- acd_only and nisp_only are distinct accounting rows over the same matched config set.",
            "- acd_only attributes shared immutable context residency and wafer-local bytes.",
            "- nisp_only attributes private parked state only; shared ACD context is excluded to avoid double-counting.",
            "- If JCT rows are equal for acd_only/nisp_only, the isolated claim is limited to model-side token and traffic accounting.",
        ]
    )
    p = Path(out_md)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", default="data/results/agentweaver_v12_policy_comparison.csv")
    ap.add_argument("--schedule", default="data/results/schedule_summary_pr4_v12.csv")
    ap.add_argument("--out", default="data/results/agentweaver_v13_mode_comparison.csv")
    ap.add_argument("--summary", default="data/results/agentweaver_v13_mode_summary.md")
    args = ap.parse_args()
    rows = build_mode_comparison(args.comparison, args.schedule, args.out, args.summary)
    print(json.dumps({"rows": len(rows), "matched_configs": rows[0]["matched_configs"] if rows else 0}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
