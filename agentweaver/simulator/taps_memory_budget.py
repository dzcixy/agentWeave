from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.analysis.context_segment_graph import kv_bytes_per_token
from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.taps_domain_scheduler import context_domain_id
from agentweaver.tracing.trace_schema import ContextSegmentRef, Event, load_trace_dir
from agentweaver.utils.io import ensure_dir, write_csv


@dataclass
class CacheEntry:
    key: str
    tokens: int
    bytes: int
    value: float
    last_access: int
    segment_type: str


def _events(trace_dirs: list[str | Path]) -> list[Event]:
    out: list[Event] = []
    for d in trace_dirs:
        if not Path(d).exists():
            continue
        for tr in load_trace_dir(d):
            out.extend(e for e in tr.events if e.node_type == "llm")
    return sorted(out, key=lambda e: (e.instance_id, e.branch_id, e.step_id, e.node_id))


def _entry_key(ev: Event, ref: ContextSegmentRef) -> str:
    domain = context_domain_id(ev)
    return f"{domain}:{ref.segment_id}"


def _criticality(ev: Event) -> float:
    if ev.context_segments and any(r.segment_type in {"task", "repo", "tool_schema"} for r in ev.context_segments):
        return 1.5
    return 1.0


def run_memory_budget(
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    out_csv: str | Path = "data/results/taps_memory_budget_pr4_v4.csv",
    budgets_gb: list[int] | None = None,
) -> list[dict[str, Any]]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    budgets_gb = budgets_gb or [4, 8, 16, 32, 64]
    events = _events(trace_dirs)
    lm = LatencyModel.load(model_json)
    bpt = kv_bytes_per_token()
    rows: list[dict[str, Any]] = []
    for budget_gb in budgets_gb:
        budget = budget_gb * 1024**3
        cache: dict[str, CacheEntry] = {}
        occupancy = 0
        hit_tokens = 0
        evicted_tokens = 0
        recompute_tokens = 0
        eviction_count = 0
        jct = 0.0
        step = 0
        for ev in events:
            step += 1
            event_hit = 0
            refs = sorted(ev.context_segments, key=lambda r: (r.start_pos, r.segment_id))
            for ref in refs:
                key = _entry_key(ev, ref)
                if key in cache:
                    entry = cache[key]
                    event_hit += ref.length
                    hit_tokens += ref.length
                    entry.last_access = step
                    entry.value += lm.predict_prefill(ref.length) * 0.2
                else:
                    recompute_tokens += ref.length
                    bytes_ = int(ref.length * bpt)
                    value = lm.predict_prefill(ref.length) * _criticality(ev) / max(1, bytes_)
                    cache[key] = CacheEntry(key, ref.length, bytes_, value, step, ref.segment_type)
                    occupancy += bytes_
            while occupancy > budget and cache:
                victim = min(cache.values(), key=lambda e: (e.value, e.last_access))
                occupancy -= victim.bytes
                evicted_tokens += victim.tokens
                eviction_count += 1
                cache.pop(victim.key, None)
            delta = max(0, ev.input_tokens - min(ev.input_tokens, event_hit))
            jct += lm.predict_prefill(delta) + lm.predict_decode(ev.context_length or ev.input_tokens, ev.output_tokens)
        rows.append(
            {
                "memory_budget_gb": budget_gb,
                "policy": "taps_memory",
                "cache_hit_tokens": hit_tokens,
                "evicted_tokens": evicted_tokens,
                "recompute_tokens": recompute_tokens,
                "memory_occupancy": occupancy,
                "eviction_count": eviction_count,
                "jct_impact": jct,
                "hit_rate": hit_tokens / max(1, hit_tokens + recompute_tokens),
            }
        )
    baseline = next((r for r in rows if int(r["memory_budget_gb"]) == max(budgets_gb)), rows[-1] if rows else {})
    base_jct = float(baseline.get("jct_impact", 0.0) or 0.0)
    for row in rows:
        row["jct_over_64gb"] = (float(row["jct_impact"]) - base_jct) / max(1e-9, base_jct)
        row["stable_gain"] = float(row["jct_over_64gb"]) <= 0.05
    write_csv(out_csv, rows)
    plot_memory(rows)
    return rows


def plot_memory(rows: list[dict[str, Any]], out: str | Path = "data/plots/taps_memory_budget_pr4_v4.pdf") -> None:
    ensure_dir(Path(out).parent)
    budgets = [int(r["memory_budget_gb"]) for r in rows]
    hit = [float(r["hit_rate"]) for r in rows]
    over = [float(r["jct_over_64gb"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
    ax1.plot(budgets, hit, marker="o", label="hit rate")
    ax1.set_xlabel("memory budget (GB)")
    ax1.set_ylabel("cache hit rate")
    ax2 = ax1.twinx()
    ax2.plot(budgets, over, marker="s", color="tab:red", label="JCT over 64GB")
    ax2.set_ylabel("JCT overhead")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/taps_memory_budget_pr4_v4.csv")
    args = ap.parse_args()
    print(json.dumps({"rows": len(run_memory_budget(out_csv=args.out)), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
