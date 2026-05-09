from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import networkx as nx

from agentweaver.tracing.dag_builder import export_dag
from agentweaver.tracing.trace_schema import ContextSegment, Event, Trace, load_trace_dir, merge_traces
from agentweaver.utils.io import ensure_dir, read_yaml, write_csv, write_json


def kv_bytes_per_token(cfg: dict[str, Any] | None = None) -> int:
    cfg = cfg or {}
    mp = cfg.get("model_params", cfg)
    return int(
        2
        * int(mp.get("num_layers", 28))
        * int(mp.get("num_kv_heads", 4))
        * int(mp.get("head_dim", 128))
        * int(mp.get("bytes_per_element", 2))
    )


def build_context_graph(events: list[Event], cfg: dict[str, Any] | None = None) -> tuple[nx.Graph, list[dict[str, Any]]]:
    bpt = kv_bytes_per_token(cfg)
    seg_defs: dict[str, ContextSegment] = {}
    consumers: dict[str, list[str]] = {}
    access_counts: dict[str, int] = {}
    for ev in events:
        if ev.node_type != "llm":
            continue
        for seg in ev.context_segment_defs:
            seg_defs.setdefault(seg.segment_id, seg)
        for ref in ev.context_segments:
            consumers.setdefault(ref.segment_id, []).append(ev.node_id)
            access_counts[ref.segment_id] = access_counts.get(ref.segment_id, 0) + 1
    g = nx.Graph()
    rows: list[dict[str, Any]] = []
    for sid, seg in seg_defs.items():
        cons = consumers.get(sid, [])
        fanout = len(set(cons))
        kvb = seg.length * bpt
        g.add_node(sid, **asdict(seg), fanout=fanout, kv_bytes=kvb, access_count=access_counts.get(sid, 0))
        for c in set(cons):
            g.add_edge(sid, c, kind="consumes")
        rows.append(
            {
                "segment_id": sid,
                "segment_type": seg.segment_type,
                "length": seg.length,
                "start_pos": seg.start_pos,
                "fanout": fanout,
                "access_count": access_counts.get(sid, 0),
                "exact_prefix_reusable": seg.exact_prefix_reusable,
                "kv_bytes": kvb,
                "repeated_prefill_tokens": max(0, fanout - 1) * seg.length,
                "exact_prefix_reusable_tokens": seg.length if seg.exact_prefix_reusable and fanout > 1 else 0,
            }
        )
    return g, rows


def process_trace_dir(trace_dir: str | Path, out: str | Path, config: str | Path | None = None) -> dict[str, Any]:
    cfg = read_yaml(config) if config else {}
    traces = load_trace_dir(trace_dir)
    outp = ensure_dir(out)
    merged = merge_traces(traces, {"source_trace_dir": str(trace_dir), "num_traces": len(traces)})
    merged.to_jsonl(outp / "events.jsonl")
    g, rows = build_context_graph(merged.events, cfg)
    write_csv(outp / "context_segments.csv", rows)
    graph = nx.node_link_data(g, edges="links")
    write_json(outp / "context_graph.json", graph)
    summaries = []
    dag_dir = ensure_dir(outp / "dags")
    for tr in traces:
        summaries.append(export_dag(tr, dag_dir))
    write_csv(outp / "dag_summary.csv", summaries)
    summary = {
        "num_traces": len(traces),
        "num_events": len(merged.events),
        "num_segments": len(rows),
        "repeated_prefill_tokens": sum(r["repeated_prefill_tokens"] for r in rows),
        "exact_prefix_reusable_tokens": sum(r["exact_prefix_reusable_tokens"] for r in rows),
    }
    write_json(outp / "context_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    print(json.dumps(process_trace_dir(args.trace_dir, args.out, args.config), indent=2))


if __name__ == "__main__":
    main()
