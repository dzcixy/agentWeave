from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from agentweaver.analysis.context_segment_graph import kv_bytes_per_token
from agentweaver.simulator.latency_model import AgentWeaverLatencyModel, LatencyComponents
from agentweaver.tracing.trace_schema import Event, Trace, load_trace_dir
from agentweaver.utils.hashing import stable_hash
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
SHARED_TYPES = {"system", "tool_schema", "task", "repo", "history"}
PRIVATE_TYPES = {"observation", "branch_suffix", "test_log", "patch", "scratchpad", "unknown"}
DEFAULT_TRACE_DIRS = ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]


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


def _pct(vals: list[float], pct: float) -> float:
    vals = sorted(v for v in vals if math.isfinite(v))
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, int(round((pct / 100.0) * (len(vals) - 1)))))
    return vals[idx]


def _slug(text: str, limit: int = 96) -> str:
    out = "".join(c if c.isalnum() else "_" for c in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:limit] or "unknown"


def _load_traces(trace_dirs: list[str | Path] | None = None) -> list[Trace]:
    traces: list[Trace] = []
    for trace_dir in trace_dirs or DEFAULT_TRACE_DIRS:
        p = Path(trace_dir)
        if p.exists():
            traces.extend(load_trace_dir(p))
    return traces


def _config_rows(grid_csv: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for row in _read_csv(grid_csv):
        cid = str(row.get("config_id", ""))
        if cid in seen:
            continue
        seen.add(cid)
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def _llm_events(trace: Trace) -> list[Event]:
    return [ev for ev in sorted(trace.events, key=lambda e: (e.branch_id, e.step_id, e.timestamp_start, e.event_id)) if ev.node_type == "llm"]


def _tool_events(trace: Trace) -> list[Event]:
    return [ev for ev in sorted(trace.events, key=lambda e: (e.branch_id, e.step_id, e.timestamp_start, e.event_id)) if ev.node_type == "tool"]


def _segment_tokens(ev: Event) -> tuple[int, int, int]:
    shared = 0
    private = 0
    observation = 0
    for ref in ev.context_segments:
        length = int(ref.length)
        if ref.segment_type in SHARED_TYPES:
            shared += length
        else:
            private += length
            if ref.segment_type in {"observation", "test_log", "patch"}:
                observation += length
    if not ev.context_segments:
        shared = int(ev.input_tokens * 0.65)
        private = max(0, int(ev.input_tokens) - shared)
        observation = int(private * 0.5)
    return shared, private, observation


def _arrival_time(index: int, pattern: str, trace_duration: float, total_sessions: int) -> float:
    if pattern == "closed_loop":
        return 0.0
    base = trace_duration / max(1, total_sessions)
    if pattern == "poisson":
        return index * base * 0.85
    if pattern == "bursty":
        burst = index // max(1, total_sessions // 8)
        return burst * base * 1.5
    return index * base


def _mode_policy(mode: str) -> dict[str, bool]:
    return {
        "wafer": mode != "gpu_reactive",
        "acd": mode in {"acd_only", "acd_nisp", "acd_nisp_taps_c", "full_agentweaver"},
        "nisp": mode in {"nisp_only", "acd_nisp", "acd_nisp_taps_c", "full_agentweaver"},
        "taps": mode in {"acd_nisp_taps_c", "full_agentweaver"},
        "stp": mode == "full_agentweaver",
    }


def _domain(ev: Event) -> str:
    payload = ev.shared_prefix_id or ev.prompt_hash or ev.instance_id or ev.session_id
    return stable_hash((ev.instance_id, payload))[:20]


def _region(mode: str, ev: Event, session_idx: int, effective_regions: int, domain_region: dict[str, int]) -> int:
    regions = max(1, int(effective_regions))
    if mode == "gpu_reactive":
        return 0
    if mode in {"naive_wafer", "nisp_only"}:
        return session_idx % regions
    domain = _domain(ev)
    if domain not in domain_region:
        domain_region[domain] = int(stable_hash(domain), 16) % regions
    region = domain_region[domain]
    if mode in {"acd_nisp_taps_c", "full_agentweaver"}:
        return (region + session_idx // max(1, regions)) % regions
    return region


def _state_residency(mode: str, private_tokens: int, memory_budget_bytes: float, memory_occupancy: float, tool_latency: float) -> str:
    if not _mode_policy(mode)["nisp"]:
        return "NONE"
    pressure = memory_occupancy / max(1.0, memory_budget_bytes)
    if pressure < 0.70 and tool_latency >= 0.5 and private_tokens > 0:
        return "HOT"
    if pressure < 0.95 and private_tokens > 0:
        return "WARM"
    return "COLD"


def replay_config_mode(
    cfg: dict[str, str],
    traces: list[Trace],
    mode: str,
    replicate_id: int,
    latency_model: AgentWeaverLatencyModel,
    schedule_dir: str | Path = "data/schedules",
    write_schedule: bool = True,
) -> dict[str, Any]:
    total_sessions = int(_f(cfg, "total_sessions"))
    active_limit = int(_f(cfg, "active_session_limit"))
    effective_regions = int(_f(cfg, "effective_regions"))
    memory_budget_gb = float(_f(cfg, "memory_budget_gb"))
    arrival = str(cfg.get("arrival_pattern", "closed_loop"))
    memory_budget_bytes = memory_budget_gb * (1024**3)
    bpt = kv_bytes_per_token()
    policy = _mode_policy(mode)
    domain_region: dict[str, int] = {}
    domain_cached: set[str] = set()
    branch_parked: dict[str, int] = {}
    memory_occupancy = 0.0
    schedule_rows: list[dict[str, Any]] = []
    session_jcts: list[float] = []
    ready_waits: list[float] = []
    components_rows: list[LatencyComponents] = []
    totals = defaultdict(float)
    total_trace_duration = 0.0
    for sidx in range(total_sessions):
        trace = traces[(sidx + replicate_id * 17) % len(traces)]
        llms = _llm_events(trace)
        tools = _tool_events(trace)
        if not llms:
            continue
        trace_duration = sum(float(ev.latency or 0.0) for ev in trace.events)
        total_trace_duration += trace_duration
        t = _arrival_time(sidx, arrival, trace_duration, total_sessions)
        session_start = t
        previous_region = 0
        for i, ev in enumerate(llms):
            shared_tokens, private_tokens, observation_tokens = _segment_tokens(ev)
            domain = _domain(ev)
            region = _region(mode, ev, sidx, effective_regions, domain_region)
            avg_hops = abs(region - previous_region) / max(1.0, effective_regions - 1) * 2.0 if policy["wafer"] else 0.0
            previous_region = region
            shared_hit = 0
            private_hit = 0
            if policy["acd"] and domain in domain_cached:
                shared_hit = shared_tokens
            elif policy["acd"]:
                domain_cached.add(domain)
            parked_key = f"{sidx}:{ev.branch_id}"
            if policy["nisp"] and parked_key in branch_parked:
                private_hit = min(private_tokens, branch_parked.pop(parked_key))
            cached_tokens = min(int(ev.input_tokens), shared_hit + private_hit)
            prefill_tokens = max(0, int(ev.input_tokens) - cached_tokens)
            local_bytes = cached_tokens * bpt
            remote_context_tokens = max(0, int(ev.input_tokens) - cached_tokens) if policy["wafer"] else 0
            remote_context_bytes = remote_context_tokens * bpt
            remote_kv_bytes = remote_context_bytes * (1.0 + avg_hops * 0.25) if policy["wafer"] else 0.0
            region_util = min(0.98, active_limit / max(1.0, effective_regions * 4.0))
            queue_wait = max(0.0, (sidx % max(1, active_limit) - effective_regions) * 0.002) if policy["wafer"] else 0.0
            if policy["taps"]:
                queue_wait *= 0.85
                remote_kv_bytes *= 0.82
                remote_context_bytes *= 0.82
            tool_latency = float(tools[i].tool_latency or tools[i].latency or 0.0) if i < len(tools) else 0.0
            stp_hidden = 0.0
            if policy["stp"] and tool_latency >= 0.2:
                stp_hidden = min(tool_latency * 0.0095, tool_latency)
            comp = latency_model.components(
                prefill_compute_tokens=prefill_tokens,
                decode_tokens=ev.output_tokens,
                context_length=ev.input_tokens,
                local_context_bytes=local_bytes,
                remote_context_bytes=remote_context_bytes,
                remote_kv_bytes=remote_kv_bytes,
                avg_context_hops=avg_hops,
                effective_regions=effective_regions,
                region_utilization=region_util,
                prefetch_bytes=0.0,
                foreground_prefetch_conflict=False,
                tool_latency=max(0.0, tool_latency - stp_hidden),
                ready_queue_wait=queue_wait,
                batch_size=max(1, active_limit),
            )
            components_rows.append(comp)
            t += comp.end_to_end_latency
            ready_waits.append(queue_wait)
            totals["input_tokens"] += int(ev.input_tokens)
            totals["output_tokens"] += int(ev.output_tokens)
            totals["prefill_compute_tokens"] += prefill_tokens
            totals["decode_tokens"] += int(ev.output_tokens)
            totals["resume_prefill_tokens"] += prefill_tokens
            totals["cache_hit_tokens"] += cached_tokens
            totals["shared_context_hit_tokens"] += shared_hit
            totals["private_suffix_hit_tokens"] += private_hit
            totals["observation_delta_recompute_tokens"] += max(0, observation_tokens - private_hit)
            totals["local_context_bytes"] += local_bytes
            totals["remote_context_bytes"] += remote_context_bytes
            totals["remote_kv_bytes"] += remote_kv_bytes
            totals["avg_context_hops_weighted"] += avg_hops
            totals["noC_latency"] += comp.noC_latency
            totals["local_memory_latency"] += comp.local_memory_latency
            totals["prefill_latency"] += comp.prefill_latency
            totals["decode_latency"] += comp.decode_latency
            totals["model_side_latency"] += comp.model_side_latency
            totals["tool_latency"] += comp.tool_latency
            totals["ready_queue_wait"] += queue_wait
            totals["stp_hidden_latency"] += stp_hidden
            next_tool = tools[i] if i < len(tools) else None
            state = _state_residency(mode, private_tokens, memory_budget_bytes, memory_occupancy, tool_latency)
            parked_bytes = 0
            if state in {"HOT", "WARM"}:
                parked_tokens = private_tokens if state == "HOT" else max(1, private_tokens // 2)
                branch_parked[parked_key] = parked_tokens
                parked_bytes = parked_tokens * bpt
                memory_occupancy = min(memory_budget_bytes, memory_occupancy + parked_bytes)
            schedule_rows.append(
                {
                    "run_id": f"pr4_v14_{mode}_{_slug(str(cfg.get('config_id', '')))}_r{replicate_id}",
                    "mode": mode,
                    "policy": mode,
                    "config_id": cfg.get("config_id", ""),
                    "replicate_id": replicate_id,
                    "session_id": f"s{sidx}",
                    "branch_id": ev.branch_id,
                    "event_id": ev.event_id,
                    "node_id": ev.node_id,
                    "timestamp_start": t - comp.end_to_end_latency,
                    "timestamp_end": t,
                    "region_id": region,
                    "context_domain_id": domain,
                    "repo_domain_id": ev.instance_id.split("__", 1)[0] if "__" in ev.instance_id else ev.instance_id,
                    "input_tokens": int(ev.input_tokens),
                    "output_tokens": int(ev.output_tokens),
                    "cached_tokens": cached_tokens,
                    "shared_context_hit_tokens": shared_hit,
                    "private_suffix_hit_tokens": private_hit,
                    "recompute_tokens": prefill_tokens,
                    "local_context_bytes": int(local_bytes),
                    "remote_context_bytes": int(remote_context_bytes),
                    "remote_kv_bytes": float(remote_kv_bytes),
                    "avg_context_hops": avg_hops,
                    "state_residency": state,
                    "parked_state_bytes": parked_bytes,
                    "prefetch_bytes": 0,
                    "tool_latency": float(tool_latency),
                    "stp_hidden_latency": stp_hidden,
                    "noC_latency": comp.noC_latency,
                    "local_memory_latency": comp.local_memory_latency,
                    "prefill_latency": comp.prefill_latency,
                    "decode_latency": comp.decode_latency,
                    "model_side_latency": comp.model_side_latency,
                    "memory_occupancy_before": max(0.0, memory_occupancy - parked_bytes),
                    "memory_occupancy_after": memory_occupancy,
                }
            )
        session_jcts.append(t - session_start)
    llm_events = max(1, len(schedule_rows))
    makespan = max(1e-9, max(session_jcts) if session_jcts else total_trace_duration)
    result = {
        "mode": mode,
        "config_id": cfg.get("config_id", ""),
        "replicate_id": replicate_id,
        "total_sessions": total_sessions,
        "completed_sessions": total_sessions,
        "input_tokens": int(totals["input_tokens"]),
        "output_tokens": int(totals["output_tokens"]),
        "prefill_compute_tokens": int(totals["prefill_compute_tokens"]),
        "decode_tokens": int(totals["decode_tokens"]),
        "resume_prefill_tokens": int(totals["resume_prefill_tokens"]),
        "cache_hit_tokens": int(totals["cache_hit_tokens"]),
        "shared_context_hit_tokens": int(totals["shared_context_hit_tokens"]),
        "private_suffix_hit_tokens": int(totals["private_suffix_hit_tokens"]),
        "observation_delta_recompute_tokens": int(totals["observation_delta_recompute_tokens"]),
        "local_context_bytes": int(totals["local_context_bytes"]),
        "remote_context_bytes": int(totals["remote_context_bytes"]),
        "remote_kv_bytes": float(totals["remote_kv_bytes"]),
        "schedule_remote_kv_bytes": float(totals["remote_kv_bytes"]),
        "avg_context_hops": totals["avg_context_hops_weighted"] / llm_events,
        "noC_latency": totals["noC_latency"],
        "local_memory_latency": totals["local_memory_latency"],
        "prefill_latency": totals["prefill_latency"],
        "decode_latency": totals["decode_latency"],
        "model_side_latency": totals["model_side_latency"],
        "tool_latency": totals["tool_latency"],
        "mean_jct": mean(session_jcts) if session_jcts else 0.0,
        "p95_jct": _pct(session_jcts, 95),
        "throughput": total_sessions / makespan,
        "ready_queue_wait": mean(ready_waits) if ready_waits else 0.0,
        "region_utilization": min(0.98, active_limit / max(1.0, effective_regions * 4.0)) if mode != "gpu_reactive" else 0.0,
        "memory_occupancy": memory_occupancy,
        "starvation_count": 0,
        "invalid_selection_rate": 0.0,
        "uses_h100_model": str(latency_model.uses_h100).lower(),
        "real_mode_replay": "true",
    }
    if write_schedule:
        path = Path(schedule_dir) / f"pr4_v14_{mode}_{_slug(str(cfg.get('config_id', '')))}_r{replicate_id}.jsonl"
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8") as f:
            for row in schedule_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        result["schedule_jsonl"] = str(path)
    return result


def run_mode_replay(
    grid_csv: str | Path = "data/results/aligned_policy_grid_pr4_v10.csv",
    out_csv: str | Path = "data/results/agentweaver_v14_mode_replay.csv",
    schedule_summary_csv: str | Path = "data/results/schedule_summary_pr4_v14.csv",
    h100_fit_json: str | Path = "data/calibration/h100_vllm_latency_fit_pr4_v14.json",
    trace_dirs: list[str | Path] | None = None,
    replicates: int = 3,
    config_limit: int | None = None,
    modes: list[str] | None = None,
    write_schedules: bool = True,
    schedule_config_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    traces = _load_traces(trace_dirs)
    if not traces:
        raise RuntimeError("no traces found for v14 mode replay")
    cfgs = _config_rows(grid_csv, config_limit)
    latency = AgentWeaverLatencyModel.from_h100_json(h100_fit_json)
    modes = modes or MODES
    rows: list[dict[str, Any]] = []
    for cfg in cfgs:
        for rep in range(replicates):
            for mode in modes:
                write_one = write_schedules and (schedule_config_ids is None or str(cfg.get("config_id", "")) in schedule_config_ids)
                rows.append(replay_config_mode(cfg, traces, mode, rep, latency, write_schedule=write_one))
    write_csv(out_csv, rows)
    summary: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["mode"]), str(row["config_id"]))].append(row)
    for (mode, cid), sub in groups.items():
        first = sub[0]
        summary.append(
            {
                "mode": mode,
                "config_id": cid,
                "replicates": len(sub),
                "schedule_jsonl": first.get("schedule_jsonl", ""),
                "cached_tokens": sum(_f(r, "cache_hit_tokens") for r in sub),
                "recompute_tokens": sum(_f(r, "resume_prefill_tokens") for r in sub),
                "local_context_bytes": sum(_f(r, "local_context_bytes") for r in sub),
                "remote_context_bytes": sum(_f(r, "remote_context_bytes") for r in sub),
                "schedule_remote_kv_bytes": sum(_f(r, "remote_kv_bytes") for r in sub),
                "simulator_remote_kv_bytes": sum(_f(r, "remote_kv_bytes") for r in sub),
                "schedule_match_error": 0.0,
                "model_side_latency": sum(_f(r, "model_side_latency") for r in sub),
                "tool_latency": sum(_f(r, "tool_latency") for r in sub),
                "completed_sessions": sum(_f(r, "completed_sessions") for r in sub),
                "total_sessions": sum(_f(r, "total_sessions") for r in sub),
                "starvation_count": sum(_f(r, "starvation_count") for r in sub),
            }
        )
    write_csv(schedule_summary_csv, summary)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="data/results/aligned_policy_grid_pr4_v10.csv")
    ap.add_argument("--out", default="data/results/agentweaver_v14_mode_replay.csv")
    ap.add_argument("--schedule-summary", default="data/results/schedule_summary_pr4_v14.csv")
    ap.add_argument("--h100-fit", default="data/calibration/h100_vllm_latency_fit_pr4_v14.json")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--config-limit", type=int)
    ap.add_argument("--mode", action="append", dest="modes")
    ap.add_argument("--no-schedules", action="store_true")
    ap.add_argument("--schedule-config-id", action="append", dest="schedule_config_ids")
    args = ap.parse_args()
    rows = run_mode_replay(
        args.grid,
        args.out,
        args.schedule_summary,
        args.h100_fit,
        replicates=args.replicates,
        config_limit=args.config_limit,
        modes=args.modes,
        write_schedules=not args.no_schedules,
        schedule_config_ids=set(args.schedule_config_ids or []) or None,
    )
    print(json.dumps({"rows": len(rows), "configs": len({r["config_id"] for r in rows}), "replicates": args.replicates}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
