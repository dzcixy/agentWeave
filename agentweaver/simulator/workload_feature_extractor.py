from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agentweaver.analysis.context_segment_graph import kv_bytes_per_token
from agentweaver.tracing.trace_schema import Trace, load_trace_dir
from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.io import ensure_dir, write_csv


DEFAULT_TRACE_DIRS = ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key)
        return default if v in ("", None) else float(v)
    except Exception:
        return default


def _percentile(vals: list[float], p: float) -> float:
    vals = sorted(v for v in vals if math.isfinite(v))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    idx = (len(vals) - 1) * p / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)


def _cv(vals: list[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    if mean <= 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    return math.sqrt(var) / mean


def _load_traces(trace_dirs: list[str | Path]) -> list[Trace]:
    traces: list[Trace] = []
    for trace_dir in trace_dirs:
        p = Path(trace_dir)
        if p.exists():
            traces.extend(load_trace_dir(p))
    return traces


def summarize_traces(trace_dirs: list[str | Path] | None = None) -> dict[str, float]:
    traces = _load_traces(trace_dirs or DEFAULT_TRACE_DIRS)
    llm_times: list[float] = []
    tool_times: list[float] = []
    branch_jcts: list[float] = []
    context_tokens = 0
    repo_tokens = 0
    history_tokens = 0
    shared_context_tokens = 0
    domain_counts: Counter[str] = Counter()
    segment_hash_counts: Counter[str] = Counter()
    total_llm_input = 0

    for trace in traces:
        by_branch: dict[str, list[Any]] = defaultdict(list)
        for ev in trace.events:
            by_branch[ev.branch_id].append(ev)
            if ev.node_type == "llm":
                dur = float(ev.latency or max(0.0, (ev.timestamp_end or 0.0) - (ev.timestamp_start or 0.0)))
                if dur > 0:
                    llm_times.append(dur)
                total_llm_input += ev.input_tokens
                domain_counts[ev.shared_prefix_id or ev.prompt_hash or ev.instance_id] += 1
                for ref in ev.context_segments:
                    context_tokens += ref.length
                    if ref.segment_type in {"system", "tool_schema", "task", "repo", "history"}:
                        shared_context_tokens += ref.length
                    if ref.segment_type == "repo":
                        repo_tokens += ref.length
                    if ref.segment_type == "history":
                        history_tokens += ref.length
                    segment_hash_counts[f"{ref.segment_type}:{ref.segment_id}:{ref.length}"] += 1
            elif ev.node_type == "tool":
                dur = float(ev.tool_latency if ev.tool_latency is not None else ev.latency or 0.0)
                if dur > 0:
                    tool_times.append(dur)
        for events in by_branch.values():
            times = [
                (float(e.timestamp_start or 0.0), float(e.timestamp_end or 0.0))
                for e in events
                if e.timestamp_start and e.timestamp_end
            ]
            if times:
                branch_jcts.append(max(t[1] for t in times) - min(t[0] for t in times))

    llm_mean = sum(llm_times) / max(1, len(llm_times))
    tool_mean = sum(tool_times) / max(1, len(tool_times))
    repeated_tokens = sum((count - 1) for count in segment_hash_counts.values() if count > 1)
    entropy = 0.0
    total_domains = sum(domain_counts.values())
    for count in domain_counts.values():
        p = count / max(1, total_domains)
        entropy -= p * math.log2(max(p, 1e-12))
    return {
        "trace_llm_time_mean": llm_mean,
        "trace_llm_time_p95": _percentile(llm_times, 95),
        "trace_tool_time_mean": tool_mean,
        "trace_tool_time_p95": _percentile(tool_times, 95),
        "tool_time_share": tool_mean / max(1e-9, tool_mean + llm_mean),
        "llm_time_share": llm_mean / max(1e-9, tool_mean + llm_mean),
        "branch_jct_cv": _cv(branch_jcts),
        "tool_latency_cv": _cv(tool_times),
        "context_reuse_tokens": float(repeated_tokens),
        "shared_context_ratio": shared_context_tokens / max(1, context_tokens),
        "repo_context_tokens": float(repo_tokens),
        "history_context_tokens": float(history_tokens),
        "estimated_kv_bytes": context_tokens * kv_bytes_per_token(),
        "estimated_domain_fanout": (sum(domain_counts.values()) / max(1, len(domain_counts))),
        "estimated_context_entropy": entropy,
        "total_llm_input_tokens": float(total_llm_input),
    }


def _repo_family(trace: Trace) -> str:
    raw = str(trace.metadata.get("repo", "") or trace.metadata.get("instance_id", "") or trace.metadata.get("source", "unknown"))
    if "__" in raw:
        return raw.split("__", 1)[0]
    if "/" in raw:
        return raw.rsplit("/", 1)[-1].split(".", 1)[0]
    return raw.split("-", 1)[0] if raw else "unknown"


def _trace_mix(traces: list[Trace], cid: str, total_sessions: int) -> list[Trace]:
    if not traces:
        return []
    ordered = sorted(traces, key=lambda t: stable_hash((cid, t.metadata.get("source", ""), t.metadata.get("instance_id", ""))))
    return [ordered[int(stable_hash((cid, i)), 16) % len(ordered)] for i in range(max(1, total_sessions))]


def summarize_trace_mix(traces: list[Trace], cid: str, total_sessions: int) -> dict[str, Any]:
    mix = _trace_mix(traces, cid, total_sessions)
    llm_times: list[float] = []
    tool_times: list[float] = []
    branch_jcts: list[float] = []
    context_tokens = 0
    shared_context_tokens = 0
    repo_tokens = 0
    history_tokens = 0
    total_llm_input = 0
    segment_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    repo_counts: Counter[str] = Counter()
    tool_events = 0
    llm_events = 0
    for trace in mix:
        repo_counts[_repo_family(trace)] += 1
        by_branch: dict[str, list[Any]] = defaultdict(list)
        for ev in trace.events:
            by_branch[ev.branch_id].append(ev)
            if ev.node_type == "llm":
                llm_events += 1
                dur = float(ev.latency or max(0.0, (ev.timestamp_end or 0.0) - (ev.timestamp_start or 0.0)))
                if dur > 0:
                    llm_times.append(dur)
                total_llm_input += ev.input_tokens
                domain = ev.shared_prefix_id or ev.prompt_hash or ev.instance_id
                domain_counts[domain] += 1
                for ref in ev.context_segments:
                    context_tokens += ref.length
                    segment_counts[f"{ref.segment_type}:{ref.segment_id}:{ref.length}"] += 1
                    if ref.segment_type in {"system", "tool_schema", "task", "repo", "history"}:
                        shared_context_tokens += ref.length
                    if ref.segment_type == "repo":
                        repo_tokens += ref.length
                    if ref.segment_type == "history":
                        history_tokens += ref.length
            elif ev.node_type == "tool":
                tool_events += 1
                dur = float(ev.tool_latency if ev.tool_latency is not None else ev.latency or 0.0)
                if dur > 0:
                    tool_times.append(dur)
        for events in by_branch.values():
            times = [
                (float(e.timestamp_start or 0.0), float(e.timestamp_end or 0.0))
                for e in events
                if e.timestamp_start and e.timestamp_end
            ]
            if times:
                branch_jcts.append(max(t[1] for t in times) - min(t[0] for t in times))
    llm_mean = sum(llm_times) / max(1, len(llm_times))
    tool_mean = sum(tool_times) / max(1, len(tool_times))
    repeated_tokens = sum((count - 1) for count in segment_counts.values() if count > 1)
    total_domains = sum(domain_counts.values())
    entropy = 0.0
    hotspot = 0.0
    if total_domains:
        hotspot = max(domain_counts.values()) / total_domains
    for count in domain_counts.values():
        p = count / max(1, total_domains)
        entropy -= p * math.log2(max(p, 1e-12))
    total_repos = sum(repo_counts.values())
    repo_dist = {repo: count / max(1, total_repos) for repo, count in sorted(repo_counts.items())}
    return {
        "trace_mix_instance_entropy": entropy,
        "repo_family_distribution": json.dumps(repo_dist, sort_keys=True),
        "trace_llm_time_mean": llm_mean,
        "trace_llm_time_p95": _percentile(llm_times, 95),
        "trace_tool_time_mean": tool_mean,
        "trace_tool_time_p95": _percentile(tool_times, 95),
        "tool_time_share": tool_mean / max(1e-9, tool_mean + llm_mean),
        "llm_time_share": llm_mean / max(1e-9, tool_mean + llm_mean),
        "branch_jct_cv": _cv(branch_jcts),
        "tool_latency_cv": _cv(tool_times),
        "context_reuse_tokens": float(repeated_tokens),
        "shared_context_ratio": shared_context_tokens / max(1, context_tokens),
        "repo_context_tokens": float(repo_tokens),
        "history_context_tokens": float(history_tokens),
        "estimated_kv_bytes": context_tokens * kv_bytes_per_token(),
        "estimated_context_domain_count": len(domain_counts),
        "estimated_domain_fanout": total_domains / max(1, len(domain_counts)),
        "estimated_context_entropy": entropy,
        "context_hotspot_score": hotspot,
        "total_llm_input_tokens": float(total_llm_input),
        "tool_events_per_session": tool_events / max(1, len(mix)),
        "llm_events_per_session": llm_events / max(1, len(mix)),
    }


def _burstiness(arrival: str) -> float:
    return {"closed_loop": 0.2, "poisson": 0.55, "bursty": 1.0}.get(arrival, 0.5)


def extract_workload_features(
    grid_csv: str | Path = "data/results/aligned_policy_grid_valid_pr4_v10.csv",
    trace_dirs: list[str | Path] | None = None,
    out_csv: str | Path = "data/results/workload_features_pr4_v10.csv",
) -> list[dict[str, Any]]:
    rows = _read_csv(grid_csv)
    seen: dict[str, dict[str, str]] = {}
    for row in rows:
        seen.setdefault(str(row.get("config_id", "")), row)
    traces = _load_traces(trace_dirs or DEFAULT_TRACE_DIRS)
    out_rows: list[dict[str, Any]] = []
    for cid, row in sorted(seen.items()):
        total = max(1.0, _f(row, "total_sessions"))
        active = max(1.0, _f(row, "active_session_limit"))
        regions = max(1.0, _f(row, "effective_regions"))
        memory = max(1.0, _f(row, "memory_budget_gb"))
        arrival = str(row.get("arrival_pattern", ""))
        trace_summary = summarize_trace_mix(traces, cid, int(total))
        session_pressure = total / active
        region_pressure = active / regions
        tool_share = trace_summary["tool_time_share"]
        context_ratio = trace_summary["shared_context_ratio"]
        memory_per_active = memory / active
        predicted_ready_depth = min(active, regions * (1.0 + (1.0 - tool_share))) * (1.0 + _burstiness(arrival) * 0.25)
        predicted_blocked_fraction = min(0.95, tool_share * (1.0 + _burstiness(arrival) * 0.25))
        estimated_remote_pressure = trace_summary["estimated_kv_bytes"] * region_pressure / max(1.0, memory * 1024**3)
        estimated_remote_context_bytes = trace_summary["estimated_kv_bytes"] * min(1.0, region_pressure / max(1.0, regions))
        feat = {
            "config_id": cid,
            "total_sessions": int(total),
            "active_session_limit": int(active),
            "effective_regions": int(regions),
            "memory_budget_gb": int(memory),
            "arrival_pattern": arrival,
            "session_pressure": session_pressure,
            "region_pressure": region_pressure,
            "memory_budget_per_active_session": memory_per_active,
            "predicted_arrival_burstiness": _burstiness(arrival),
            "trace_mix_instance_entropy": trace_summary["trace_mix_instance_entropy"],
            "repo_family_distribution": trace_summary["repo_family_distribution"],
            "trace_llm_time_mean": trace_summary["trace_llm_time_mean"],
            "trace_llm_time_p95": trace_summary["trace_llm_time_p95"],
            "trace_tool_time_mean": trace_summary["trace_tool_time_mean"],
            "trace_tool_time_p95": trace_summary["trace_tool_time_p95"],
            "tool_time_share": tool_share,
            "llm_time_share": trace_summary["llm_time_share"],
            "branch_jct_cv": trace_summary["branch_jct_cv"],
            "tool_latency_cv": trace_summary["tool_latency_cv"],
            "context_reuse_tokens": trace_summary["context_reuse_tokens"],
            "shared_context_ratio": context_ratio,
            "repo_context_tokens": trace_summary["repo_context_tokens"],
            "history_context_tokens": trace_summary["history_context_tokens"],
            "estimated_kv_bytes": trace_summary["estimated_kv_bytes"],
            "estimated_domain_fanout": trace_summary["estimated_domain_fanout"],
            "estimated_context_entropy": trace_summary["estimated_context_entropy"],
            "estimated_context_domain_count": trace_summary["estimated_context_domain_count"],
            "context_hotspot_score": trace_summary["context_hotspot_score"],
            "tool_events_per_session": trace_summary["tool_events_per_session"],
            "llm_events_per_session": trace_summary["llm_events_per_session"],
            "predicted_ready_depth": predicted_ready_depth,
            "predicted_blocked_fraction": predicted_blocked_fraction,
            "predicted_tool_blocked_fraction": predicted_blocked_fraction,
            "estimated_remote_kv_pressure": estimated_remote_pressure,
            "estimated_remote_context_bytes": estimated_remote_context_bytes,
            "feature_sources_json": json.dumps(
                {
                    "config": [
                        "total_sessions",
                        "active_session_limit",
                        "effective_regions",
                        "memory_budget_gb",
                        "arrival_pattern",
                    ],
                    "trace": [
                        "trace_llm_time_mean",
                        "trace_tool_time_mean",
                        "trace_mix_instance_entropy",
                        "repo_family_distribution",
                        "context_reuse_tokens",
                        "shared_context_ratio",
                    ],
                    "predicted": [
                        "predicted_arrival_burstiness",
                        "predicted_ready_depth",
                        "predicted_blocked_fraction",
                        "predicted_tool_blocked_fraction",
                        "estimated_remote_kv_pressure",
                        "estimated_remote_context_bytes",
                    ],
                    "policy_outcome": [],
                },
                sort_keys=True,
            ),
        }
        out_rows.append(feat)
    write_csv(out_csv, out_rows)
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="data/results/aligned_policy_grid_valid_pr4_v10.csv")
    ap.add_argument("--trace-dir", action="append", dest="trace_dirs")
    ap.add_argument("--out", default="data/results/workload_features_pr4_v10.csv")
    args = ap.parse_args()
    rows = extract_workload_features(args.grid, args.trace_dirs, args.out)
    print({"rows": len(rows), "out": args.out})


if __name__ == "__main__":
    main()
