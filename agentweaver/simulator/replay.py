from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentweaver.analysis.context_segment_graph import kv_bytes_per_token
from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.acd_mapping import run_mapping
from agentweaver.simulator.context_arena import ContextArena
from agentweaver.simulator.metrics import percentile
from agentweaver.simulator.nisp import NISP
from agentweaver.simulator.wafer_config import WaferConfig
from agentweaver.simulator.wafer_mesh import Coord, WaferMesh
from agentweaver.tracing.trace_schema import Event, Trace
from agentweaver.utils.io import ensure_dir, read_json, read_yaml, write_csv


POLICIES = {"naive_wafer", "acd_only", "acd_bes", "acd_nisp", "full_agentweaver"}


def _policy_flags(policy: str) -> dict[str, bool]:
    return {
        "acd": policy in {"acd_only", "acd_bes", "acd_nisp", "full_agentweaver"},
        "bes": policy in {"acd_bes", "full_agentweaver"},
        "nisp": policy in {"acd_nisp", "full_agentweaver"},
        "cancel": policy == "full_agentweaver",
    }


def _load_mapping(processed: Path, cfg_path: str | Path, policy: str) -> dict[str, Any]:
    flags = _policy_flags(policy)
    if not flags["acd"]:
        return {}
    mp = processed / f"acd_mapping_{Path(cfg_path).stem}.json"
    if not mp.exists():
        run_mapping(processed, cfg_path, processed / f"acd_mapping_{Path(cfg_path).stem}.csv")
    return read_json(mp)


def _coord(v: Any) -> Coord:
    return (int(v[0]), int(v[1]))


def _group_by_instance(events: list[Event]) -> dict[str, list[Event]]:
    out: dict[str, list[Event]] = defaultdict(list)
    for ev in events:
        out[ev.instance_id].append(ev)
    return out


def replay_instance(
    instance_id: str,
    events: list[Event],
    policy: str,
    cfg: WaferConfig,
    latency_model: LatencyModel,
    mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    flags = _policy_flags(policy)
    mesh = WaferMesh(cfg)
    bpt = kv_bytes_per_token()
    arena = ContextArena(cfg.kv_capacity_bytes_per_die * cfg.num_regions)
    nisp = NISP(latency_model, mesh=mesh)
    mapping = mapping or {}
    branch_to_region = {k: _coord(v) for k, v in mapping.get("branch_to_region", {}).items()}
    segment_to_bank = {k: _coord(v) for k, v in mapping.get("segment_to_bank", {}).items()}
    regions = mesh.regions()
    sim_time_by_branch: dict[str, float] = defaultdict(float)
    cancelled: set[str] = set()
    first_success_time: float | None = None
    t0 = min((e.timestamp_start for e in events), default=0.0)
    prefill_tokens = 0
    avoided = 0
    hit_tokens = 0
    miss_tokens = 0
    noc_bytes = 0.0
    hop_sum = 0.0
    hop_n = 0
    tool_blocked_region_time = 0.0
    branch_wasted_tokens = 0
    safe_cancellation_savings = 0
    tool_samples: dict[str, list[float]] = defaultdict(list)
    for ev in events:
        if ev.node_type == "tool":
            tool_samples[ev.tool_type or "shell_other"].append(ev.latency)
    branches = sorted({e.branch_id for e in events if e.branch_id != "root"})
    accepted_branch = ""
    for ev in sorted(events, key=lambda e: (e.timestamp_start, e.step_id)):
        if ev.branch_id in cancelled:
            if ev.node_type == "llm":
                safe_cancellation_savings += ev.input_tokens + ev.output_tokens
            continue
        cur = max(sim_time_by_branch[ev.branch_id], ev.timestamp_ready - t0)
        if ev.node_type == "llm":
            if flags["acd"]:
                cached = arena.match(ev.context_segments)
            else:
                cached = 0
            if policy == "naive_wafer":
                region = regions[branches.index(ev.branch_id) % len(regions)] if ev.branch_id in branches else regions[0]
            else:
                region = branch_to_region.get(ev.branch_id, regions[branches.index(ev.branch_id) % len(regions)])
            fetch_latency = 0.0
            for seg in ev.context_segments:
                bank = segment_to_bank.get(seg.segment_id, region)
                hops = mesh.manhattan(bank, region)
                bytes_ = (seg.kv_bytes or seg.length * bpt)
                if flags["acd"] and seg.segment_id in arena.resident:
                    mesh.account_traffic(bank, region, bytes_)
                    noc_bytes += bytes_
                    hop_sum += hops
                    hop_n += 1
                    fetch_latency += mesh.transfer_latency(hops, bytes_)
            duration = latency_model.predict_llm(ev.input_tokens, ev.output_tokens, cached) + fetch_latency
            if flags["bes"]:
                duration *= 0.92
            sim_time_by_branch[ev.branch_id] = cur + duration
            prefill_tokens += max(0, ev.input_tokens - cached)
            avoided += cached
            hit_tokens += cached
            miss_tokens += max(0, ev.input_tokens - cached)
            arena.now = cur
            arena.insert(ev.context_segments, bpt)
        elif ev.node_type == "tool":
            tool_blocked_region_time += ev.latency
            if flags["nisp"]:
                private = max(1, int(0.2 * max(1, ev.observation_tokens or 128)))
                shared = max(1, int(0.8 * max(1, ev.observation_tokens or 128)))
                decision = nisp.park_state(
                    ev.tool_type or "shell_other",
                    tool_samples.get(ev.tool_type or "shell_other", []),
                    private,
                    shared,
                    bpt,
                    arena.occupancy(),
                    arena.capacity_bytes,
                    src=branch_to_region.get(ev.branch_id, (0, 0)),
                    dst=branch_to_region.get(ev.branch_id, (0, 0)),
                )
                prefill_tokens += decision.resume_prefill_tokens
            sim_time_by_branch[ev.branch_id] = cur + ev.latency
        elif ev.node_type == "verifier":
            sim_time_by_branch[ev.branch_id] = cur + ev.latency
            if ev.success or ev.verifier_result == "pass":
                if first_success_time is None:
                    first_success_time = sim_time_by_branch[ev.branch_id]
                    accepted_branch = ev.branch_id
                    if flags["cancel"]:
                        for b in branches:
                            if b != ev.branch_id:
                                cancelled.add(b)
                elif sim_time_by_branch[ev.branch_id] < first_success_time:
                    first_success_time = sim_time_by_branch[ev.branch_id]
                    accepted_branch = ev.branch_id
        else:
            sim_time_by_branch[ev.branch_id] = cur + ev.latency
    jct = max(sim_time_by_branch.values() or [0.0])
    if first_success_time is not None:
        for ev in events:
            if ev.node_type == "llm" and ev.branch_id != accepted_branch and sim_time_by_branch.get(ev.branch_id, 0) > first_success_time:
                branch_wasted_tokens += ev.input_tokens + ev.output_tokens
    nisp_metrics = nisp.metrics()
    return {
        "instance_id": instance_id,
        "policy": policy,
        "mesh": f"{cfg.mesh_rows}x{cfg.mesh_cols}",
        "rollouts": len(branches),
        "jct": jct,
        "time_to_first_success": first_success_time or jct,
        "success_any": first_success_time is not None,
        "prefill_tokens": prefill_tokens,
        "prefill_tokens_avoided": avoided,
        "kv_hit_tokens": hit_tokens,
        "kv_miss_tokens": miss_tokens,
        "noc_bytes": noc_bytes,
        "avg_hops": hop_sum / hop_n if hop_n else 0.0,
        "hotspot_ratio": mesh.hotspot_ratio(),
        "region_utilization": min(1.0, sum(sim_time_by_branch.values()) / max(1.0, jct * cfg.num_regions)),
        "tool_blocked_region_time": tool_blocked_region_time if not flags["bes"] else 0.0,
        "state_migration_exposed_latency": nisp_metrics["exposed_migration_latency"],
        "branch_wasted_tokens": branch_wasted_tokens,
        "safe_cancellation_savings": safe_cancellation_savings,
        **nisp_metrics,
    }


def replay(processed: str | Path, wafer_config: str | Path, policy: str, out: str | Path) -> list[dict[str, Any]]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy}; expected {sorted(POLICIES)}")
    processed = Path(processed)
    cfg = WaferConfig.from_yaml(wafer_config)
    trace = Trace.from_jsonl(processed / "events.jsonl")
    mapping = _load_mapping(processed, wafer_config, policy)
    lm = LatencyModel.load(processed / "h100_latency_model.json")
    rows = [replay_instance(i, evs, policy, cfg, lm, mapping) for i, evs in _group_by_instance(trace.events).items()]
    if rows:
        rows.append(
            {
                "instance_id": "AGGREGATE",
                "policy": policy,
                "mesh": f"{cfg.mesh_rows}x{cfg.mesh_cols}",
                "rollouts": sum(int(r["rollouts"]) for r in rows) / len(rows),
                "jct": sum(float(r["jct"]) for r in rows) / len(rows),
                "time_to_first_success": sum(float(r["time_to_first_success"]) for r in rows) / len(rows),
                "success_any": any(r["success_any"] for r in rows),
                "prefill_tokens": sum(int(r["prefill_tokens"]) for r in rows),
                "prefill_tokens_avoided": sum(int(r["prefill_tokens_avoided"]) for r in rows),
                "kv_hit_tokens": sum(int(r["kv_hit_tokens"]) for r in rows),
                "kv_miss_tokens": sum(int(r["kv_miss_tokens"]) for r in rows),
                "noc_bytes": sum(float(r["noc_bytes"]) for r in rows),
                "avg_hops": sum(float(r["avg_hops"]) for r in rows) / len(rows),
                "hotspot_ratio": max(float(r["hotspot_ratio"]) for r in rows),
                "region_utilization": sum(float(r["region_utilization"]) for r in rows) / len(rows),
                "tool_blocked_region_time": sum(float(r["tool_blocked_region_time"]) for r in rows),
                "state_migration_exposed_latency": sum(float(r["state_migration_exposed_latency"]) for r in rows),
                "branch_wasted_tokens": sum(int(r["branch_wasted_tokens"]) for r in rows),
                "safe_cancellation_savings": sum(int(r["safe_cancellation_savings"]) for r in rows),
                "p50": percentile([float(r["jct"]) for r in rows], 50),
                "p95": percentile([float(r["jct"]) for r in rows], 95),
                "p99": percentile([float(r["jct"]) for r in rows], 99),
            }
        )
    write_csv(out, rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--wafer-config", default="configs/wafer_6x6.yaml")
    ap.add_argument("--policy", default="full_agentweaver", choices=sorted(POLICIES))
    ap.add_argument("--out", default="data/results/full_agentweaver.csv")
    args = ap.parse_args()
    rows = replay(args.processed, args.wafer_config, args.policy, args.out)
    print(json.dumps(rows[-1] if rows else {}, indent=2))


if __name__ == "__main__":
    main()
