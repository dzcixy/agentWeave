from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.context_arena import ContextArena
from agentweaver.analysis.context_segment_graph import kv_bytes_per_token
from agentweaver.tracing.trace_schema import Event, load_trace_dir
from agentweaver.utils.io import ensure_dir, write_csv


def _event_start(ev: Event) -> float:
    return ev.timestamp_start or ev.timestamp_ready or 0.0


def _event_end(ev: Event) -> float:
    return ev.timestamp_end or _event_start(ev)


def _branch_measured(events: list[Event]) -> dict[str, float]:
    timed = [e for e in events if not e.timing_missing and _event_start(e) and _event_end(e)]
    if timed:
        measured_jct = max(_event_end(e) for e in timed) - min(_event_start(e) for e in timed)
    else:
        measured_jct = 0.0
    llm_time = sum(float(e.latency or 0.0) for e in timed if e.node_type == "llm")
    tool_time = sum(float(e.tool_latency if e.tool_latency is not None else e.latency or 0.0) for e in timed if e.node_type == "tool")
    other = max(0.0, measured_jct - llm_time - tool_time)
    return {
        "measured_agent_jct": measured_jct,
        "measured_llm_wall_time": llm_time,
        "measured_tool_wall_time": tool_time,
        "measured_orchestration_or_other_time": other,
        "tool_time_share": tool_time / measured_jct if measured_jct > 0 else 0.0,
        "llm_time_share": llm_time / measured_jct if measured_jct > 0 else 0.0,
    }


def _decode_sum(lm: LatencyModel, events: list[Event]) -> float:
    return sum(lm.predict_decode(e.context_length or e.input_tokens, e.output_tokens) for e in events if e.node_type == "llm")


def build_breakdown(
    trace_dir: str | Path = "data/traces/mini_swe_lite10_r4_timed",
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    out_csv: str | Path = "data/results/mini_swe_lite10_r4_timed_latency_breakdown_detailed.csv",
    plot_out: str | Path = "data/plots/mini_swe_lite10_r4_timed_model_tool_breakdown_pr3_v3.pdf",
) -> list[dict[str, Any]]:
    lm = LatencyModel.load(model_json)
    bpt = kv_bytes_per_token()
    traces = load_trace_dir(trace_dir)
    by_instance: dict[str, list[Event]] = defaultdict(list)
    for trace in traces:
        for ev in trace.events:
            by_instance[ev.instance_id].append(ev)
    rows: list[dict[str, Any]] = []
    for instance_id, events in sorted(by_instance.items()):
        arena = ContextArena(80 * (1024**3) * 36)
        branch_prefill_full: dict[str, float] = defaultdict(float)
        branch_cached_tokens: dict[str, int] = defaultdict(int)
        llm_events = sorted([e for e in events if e.node_type == "llm"], key=lambda e: (e.step_id, e.branch_id, _event_start(e)))
        for ev in llm_events:
            cached = arena.match(ev.context_segments)
            branch_cached_tokens[ev.branch_id] += cached
            branch_prefill_full[ev.branch_id] += lm.predict_prefill(max(0, ev.input_tokens - cached))
            arena.insert(ev.context_segments, bpt)
        by_branch: dict[str, list[Event]] = defaultdict(list)
        for ev in events:
            if ev.branch_id != "root":
                by_branch[ev.branch_id].append(ev)
        for branch_id, branch_events in sorted(by_branch.items()):
            branch_llm = [e for e in branch_events if e.node_type == "llm"]
            measured = _branch_measured(branch_events)
            naive_prefill = sum(lm.predict_prefill(e.input_tokens) for e in branch_llm)
            full_prefill = branch_prefill_full.get(branch_id, naive_prefill)
            decode = _decode_sum(lm, branch_llm)
            rows.append(
                {
                    "instance_id": instance_id,
                    "branch_id": branch_id,
                    **measured,
                    "simulated_naive_prefill_time": naive_prefill,
                    "simulated_full_prefill_time": full_prefill,
                    "simulated_decode_time": decode,
                    "simulated_context_reuse_savings": max(0.0, naive_prefill - full_prefill),
                    "simulated_cached_tokens": branch_cached_tokens.get(branch_id, 0),
                    "policy": "naive_vs_full_agentweaver",
                }
            )
    write_csv(out_csv, rows)
    _plot(rows, plot_out)
    return rows


def _plot(rows: list[dict[str, Any]], out: str | Path) -> None:
    ensure_dir(Path(out).parent)
    total_jct = sum(float(r["measured_agent_jct"]) for r in rows)
    measured = [
        sum(float(r["measured_llm_wall_time"]) for r in rows),
        sum(float(r["measured_tool_wall_time"]) for r in rows),
        sum(float(r["measured_orchestration_or_other_time"]) for r in rows),
    ]
    simulated = [
        sum(float(r["simulated_naive_prefill_time"]) for r in rows),
        sum(float(r["simulated_full_prefill_time"]) for r in rows),
        sum(float(r["simulated_decode_time"]) for r in rows),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(["LLM", "tool", "other"], measured, color=["#4c78a8", "#f58518", "#bab0ac"])
    axes[0].set_ylabel("measured wall time (s)")
    axes[0].set_title(f"Measured agent runtime, total JCT={total_jct:.1f}s")
    axes[1].bar(["naive prefill", "full prefill", "decode"], simulated, color=["#e45756", "#72b7b2", "#54a24b"])
    axes[1].set_ylabel("simulated model-side time (s)")
    axes[1].set_title("H100 model-side replay components")
    fig.savefig(out)
    plt.close(fig)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    jct = sum(float(r["measured_agent_jct"]) for r in rows)
    tool = sum(float(r["measured_tool_wall_time"]) for r in rows)
    llm = sum(float(r["measured_llm_wall_time"]) for r in rows)
    savings = sum(float(r["simulated_context_reuse_savings"]) for r in rows)
    return {
        "MODEL_TOOL_BREAKDOWN": "PASS" if rows else "FAIL",
        "num_branch_rows": len(rows),
        "tool_time_share": tool / jct if jct > 0 else 0.0,
        "llm_time_share": llm / jct if jct > 0 else 0.0,
        "simulated_context_reuse_savings": savings,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default="data/traces/mini_swe_lite10_r4_timed")
    ap.add_argument("--model-json", default="data/profiles/h100_latency_model_pr2_v2.json")
    ap.add_argument("--out", default="data/results/mini_swe_lite10_r4_timed_latency_breakdown_detailed.csv")
    ap.add_argument("--plot-out", default="data/plots/mini_swe_lite10_r4_timed_model_tool_breakdown_pr3_v3.pdf")
    args = ap.parse_args()
    rows = build_breakdown(args.trace_dir, args.model_json, args.out, args.plot_out)
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
