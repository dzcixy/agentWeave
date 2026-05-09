from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentweaver.analysis.context_segment_graph import kv_bytes_per_token
from agentweaver.baselines.continuum_like import ContinuumLikeCache
from agentweaver.baselines.kvflow_like import KVFlowLikeCache
from agentweaver.baselines.lru_cache import LRUCache
from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.tracing.trace_schema import Event, Trace
from agentweaver.utils.io import read_yaml, write_csv


def _future_scores(events: list[Event]) -> dict[str, float]:
    positions: dict[str, list[int]] = {}
    for i, ev in enumerate(events):
        if ev.node_type == "llm":
            for seg in ev.context_segments:
                positions.setdefault(seg.segment_id, []).append(i)
    scores: dict[str, float] = {}
    for sid, pos in positions.items():
        scores[sid] = sum(1.0 / (1 + p) for p in pos)
    return scores


def simulate(processed: str | Path, config: str | Path, out: str | Path) -> list[dict[str, object]]:
    cfg = read_yaml(config)
    capacity_gb = float(cfg.get("kv_cache_capacity_gb", cfg.get("die_memory_capacity_gb", 80) * cfg.get("kv_budget_ratio", 0.3)))
    capacity = int(capacity_gb * 1024**3)
    trace = Trace.from_jsonl(Path(processed) / "events.jsonl")
    lm = LatencyModel.load(Path(processed) / "h100_latency_model.json")
    bpt = kv_bytes_per_token()
    policies = {
        "GPU-LRU": LRUCache(capacity),
        "GPU-KVFlow-like": KVFlowLikeCache(capacity, _future_scores(trace.events)),
        "GPU-Continuum-like": ContinuumLikeCache(capacity, ttl_seconds=30),
    }
    rows: list[dict[str, object]] = []
    for name, cache in policies.items():
        simulated_jct = 0.0
        prefill = avoided = hit = miss = resume = tool_resume_delay = 0
        for ev in sorted(trace.events, key=lambda e: e.timestamp_start):
            if ev.node_type == "llm":
                cached = 0
                for seg in ev.context_segments:
                    size = seg.kv_bytes or seg.length * bpt
                    if getattr(cache, "get")(seg.segment_id):
                        cached += seg.length
                    if isinstance(cache, ContinuumLikeCache):
                        cache.put_at(seg.segment_id, size, ev.timestamp_start)
                    else:
                        cache.put(seg.segment_id, size)
                prefill += max(0, ev.input_tokens - cached)
                avoided += cached
                hit += cached
                miss += max(0, ev.input_tokens - cached)
                simulated_jct += lm.predict_llm(ev.input_tokens, ev.output_tokens, cached)
            elif ev.node_type == "tool":
                if isinstance(cache, ContinuumLikeCache):
                    cache.expire(ev.timestamp_end)
                    tool_resume_delay += 0
                simulated_jct += ev.latency
        rows.append(
            {
                "policy": name,
                "simulated_jct": simulated_jct,
                "prefill_tokens": prefill,
                "prefill_tokens_avoided": avoided,
                "kv_hit_tokens": hit,
                "kv_miss_tokens": miss,
                "evictions": getattr(cache, "evictions", 0),
                "cache_occupancy": getattr(cache, "occupancy")(),
                "resume_recompute_tokens": resume,
                "tool_resume_delay": tool_resume_delay,
            }
        )
    write_csv(out, rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--config", default="configs/swebench_lite.yaml")
    ap.add_argument("--out", default="data/results/gpu_cache.csv")
    args = ap.parse_args()
    print(json.dumps(simulate(args.processed, args.config, args.out), indent=2))


if __name__ == "__main__":
    main()
