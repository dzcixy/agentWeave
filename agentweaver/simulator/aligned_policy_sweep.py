from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.taps_unified import TAPSUnifiedReplay, _load_traces
from agentweaver.simulator.taps_unified_adaptive import ADAPTIVE_POLICY, AdaptiveProfiles, TAPSAdaptiveReplay
from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.io import ensure_dir, write_csv


ALIGNED_POLICIES = [
    "reactive_admission",
    "acd_nisp",
    "taps_domain_v4",
    "taps_admission_v4",
    "taps_unified_v5",
    "taps_unified_adaptive_v6",
]


def config_id(total: int, limit: int, regions: int, arrival: str, memory: int) -> str:
    return f"aligned_v8:ts{total}:al{limit}:er{regions}:arr{arrival}:mem{memory}"


def _all_configs() -> list[tuple[int, int, int, str, int]]:
    configs: list[tuple[int, int, int, str, int]] = []
    for total, limit, regions, arrival, mem in itertools.product(
        [16, 32, 64, 128],
        [4, 8, 16, 32],
        [1, 2, 4, 8, 16],
        ["closed_loop", "poisson", "bursty"],
        [8, 16, 32, 64],
    ):
        if limit <= total:
            configs.append((total, limit, regions, arrival, mem))
    return configs


def select_configs(size: str = "medium") -> list[tuple[int, int, int, str, int]]:
    full = _all_configs()
    if size == "full":
        return full
    limit = 24 if size == "small" else 120
    must_have = {
        (16, 4, 1, "poisson", 8),
        (32, 8, 2, "closed_loop", 16),
        (64, 16, 4, "bursty", 32),
        (64, 16, 8, "poisson", 32),
        (128, 32, 16, "bursty", 64),
    }
    ordered = sorted(full, key=lambda c: stable_hash(c))
    selected: list[tuple[int, int, int, str, int]] = []
    for cfg in sorted(must_have):
        if cfg in full and cfg not in selected:
            selected.append(cfg)
    for cfg in ordered:
        if cfg not in selected:
            selected.append(cfg)
        if len(selected) >= limit:
            break
    return selected


def _run_policy(
    traces: list[Any],
    lm: LatencyModel,
    total: int,
    limit: int,
    regions: int,
    arrival: str,
    memory: int,
    policy: str,
    profiles: AdaptiveProfiles | None = None,
) -> dict[str, Any]:
    seed = 1701 + total + limit * 3 + regions * 7 + memory
    if policy == ADAPTIVE_POLICY:
        return TAPSAdaptiveReplay(traces, total, limit, regions, arrival, memory, lm, profiles=profiles, seed=seed).run()
    mapped = "taps_unified" if policy == "taps_unified_v5" else policy
    row = TAPSUnifiedReplay(traces, total, limit, regions, arrival, memory, mapped, lm, seed=seed).run()
    row["policy"] = policy
    return row


def run_aligned_policy_grid(
    size: str = "medium",
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    out_csv: str | Path = "data/results/aligned_policy_grid_pr4_v8.csv",
    missing_out: str | Path = "data/results/aligned_policy_grid_missing_pr4_v8.md",
) -> list[dict[str, Any]]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    traces = _load_traces(trace_dirs)
    lm = LatencyModel.load(model_json)
    profiles = AdaptiveProfiles.default()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for total, limit, regions, arrival, memory in select_configs(size):
        cid = config_id(total, limit, regions, arrival, memory)
        for policy in ALIGNED_POLICIES:
            try:
                row = _run_policy(traces, lm, total, limit, regions, arrival, memory, policy, profiles)
                rows.append(
                    {
                        "config_id": cid,
                        "total_sessions": total,
                        "active_session_limit": limit,
                        "effective_regions": regions,
                        "arrival_pattern": arrival,
                        "memory_budget_gb": memory,
                        "policy": policy,
                        "throughput": row.get("throughput", 0.0),
                        "mean_jct": row.get("mean_jct", 0.0),
                        "p95_jct": row.get("p95_jct", 0.0),
                        "p99_jct": row.get("p99_jct", 0.0),
                        "ready_queue_wait": row.get("ready_queue_wait", 0.0),
                        "region_utilization": row.get("region_utilization", 0.0),
                        "domain_cache_hit_rate": row.get("domain_cache_hit_rate", 0.0),
                        "blocked_session_fraction": row.get("blocked_session_fraction", 0.0),
                        "remote_kv_bytes": row.get("remote_kv_bytes", 0.0),
                        "memory_occupancy": row.get("memory_occupancy", 0.0),
                        "starvation_count": row.get("starvation_count", 0),
                        "completed_sessions": row.get("completed_sessions", 0),
                    }
                )
            except Exception as exc:
                missing.append(f"{cid},{policy},{type(exc).__name__}:{exc}")
    write_csv(out_csv, rows)
    lines = [
        "# Aligned Policy Grid Missing Rows PR4-v8",
        "",
        f"GRID_SIZE = {size}",
        f"CONFIGS_REQUESTED = {len(select_configs(size))}",
        f"ROWS_WRITTEN = {len(rows)}",
        f"MISSING_ROWS = {len(missing)}",
        "",
    ]
    lines.extend(missing or ["none"])
    p = Path(missing_out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["small", "medium", "full"], default="medium")
    ap.add_argument("--out", default="data/results/aligned_policy_grid_pr4_v8.csv")
    args = ap.parse_args()
    rows = run_aligned_policy_grid(size=args.size, out_csv=args.out)
    configs = len({r["config_id"] for r in rows})
    print(json.dumps({"rows": len(rows), "configs": configs, "out": args.out}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
