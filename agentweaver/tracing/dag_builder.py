from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import networkx as nx

from agentweaver.tracing.trace_schema import Event, Trace, load_trace_dir
from agentweaver.utils.io import ensure_dir, write_json


def build_agent_dag(events: list[Event]) -> nx.DiGraph:
    g = nx.DiGraph()
    by_branch: dict[str, list[Event]] = {}
    for ev in events:
        g.add_node(ev.node_id, **_graphml_attrs({k: v for k, v in asdict(ev).items() if k != "context_segment_defs"}))
        by_branch.setdefault(ev.branch_id, []).append(ev)
        if ev.parent_branch_id and ev.parent_branch_id != ev.branch_id:
            parent = f"{ev.instance_id}:fork" if ev.parent_branch_id == "root" else ev.parent_branch_id
            if parent in g:
                g.add_edge(parent, ev.node_id, kind="fork")
        if ev.node_type == "llm":
            for seg in ev.context_segments:
                seg_node = f"seg:{seg.segment_id}"
                if not g.has_node(seg_node):
                    g.add_node(seg_node, node_type="context_segment", segment_id=seg.segment_id, length=seg.length)
                g.add_edge(seg_node, ev.node_id, kind="context", tokens=seg.length)
    for branch_events in by_branch.values():
        ordered = sorted(branch_events, key=lambda e: (e.step_id, e.timestamp_start))
        for a, b in zip(ordered, ordered[1:]):
            g.add_edge(a.node_id, b.node_id, kind="control", latency=b.timestamp_start - a.timestamp_end)
    return g


def _graphml_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for k, v in attrs.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = json.dumps(v, sort_keys=True)
    return clean


def _event_duration(ev: Event) -> float:
    return max(0.0, ev.timestamp_end - ev.timestamp_start)


def analyze_dag(events: list[Event]) -> dict[str, Any]:
    branches = sorted({e.branch_id for e in events if e.branch_id != "root"})
    end_times = [e.timestamp_end for e in events]
    start_times = [e.timestamp_start for e in events]
    first_success = next((e for e in sorted(events, key=lambda x: x.timestamp_end) if e_success(e)), None)
    first_success_time = first_success.timestamp_end - min(start_times) if first_success and start_times else None
    wasted_tokens = 0
    if first_success:
        for ev in events:
            if ev.node_type == "llm" and ev.timestamp_start > first_success.timestamp_end and ev.branch_id != first_success.branch_id:
                wasted_tokens += ev.input_tokens + ev.output_tokens
    tool_blocked = sum(_event_duration(e) for e in events if e.node_type == "tool")
    llm_time = sum(_event_duration(e) for e in events if e.node_type == "llm")
    repeated_prefill = sum(e.input_tokens for e in events if e.node_type == "llm") - sum(
        {seg.segment_id: seg.length for e in events for seg in e.context_segments}.values()
    )
    return {
        "critical_path_length": max(end_times) - min(start_times) if end_times and start_times else 0.0,
        "success_critical_path": first_success_time or 0.0,
        "first_success_branch": first_success.branch_id if first_success else "",
        "wasted_sibling_work_after_first_success": wasted_tokens,
        "branch_join_wait": max(0.0, (max(end_times) - first_success.timestamp_end) if first_success and end_times else 0.0),
        "tool_blocked_duration": tool_blocked,
        "llm_duration": llm_time,
        "repeated_prefill_on_critical_path": max(0, repeated_prefill),
        "branch_fanout": len(branches),
    }


def e_success(ev: Event) -> bool:
    return ev.node_type == "verifier" and (ev.success is True or ev.verifier_result == "pass")


def export_dag(trace: Trace, out_dir: str | Path) -> dict[str, Any]:
    out = ensure_dir(out_dir)
    g = build_agent_dag(trace.events)
    inst = trace.events[0].instance_id if trace.events else "empty"
    nx.write_graphml(g, out / f"agent_dag_{inst}.graphml")
    data = nx.node_link_data(g, edges="links")
    write_json(out / f"agent_dag_{inst}.json", data)
    summary = analyze_dag(trace.events)
    write_json(out / f"agent_dag_{inst}_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir")
    ap.add_argument("--trace")
    ap.add_argument("--out", default="data/processed/dag")
    args = ap.parse_args()
    traces = load_trace_dir(args.trace_dir) if args.trace_dir else [Trace.from_jsonl(args.trace)]
    summaries = [export_dag(t, args.out) for t in traces]
    print(json.dumps({"num_traces": len(traces), "summaries": summaries[:3]}, indent=2))


if __name__ == "__main__":
    main()
