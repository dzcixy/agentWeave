from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.analysis.context_segment_graph import kv_bytes_per_token
from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.tracing.trace_schema import ContextSegment, Event, Trace, load_trace_dir
from agentweaver.utils.io import ensure_dir, write_csv, write_json


SHARED_INVARIANT = {"system", "tool_schema", "task", "repo"}
SHARED_APPENDABLE = {"history"}
BRANCH_DELTA = {"observation", "patch", "test_log", "branch_suffix", "scratchpad"}


@dataclass
class CDFSegment:
    segment_id: str
    segment_type: str
    segment_class: str
    instance_id: str
    length: int
    token_hash: str
    start_pos: int
    exact_prefix_reusable: bool
    fanout: int
    access_count: int
    consumer_branch_ids: list[str]
    consumer_llm_node_ids: list[str]
    prefill_cost: float
    kv_bytes: int
    utility: float
    cost: float
    resident_score: float
    cdf_selected: bool
    context_domain_id: str


def segment_class(segment_type: str, mutable: bool = False) -> str:
    if segment_type in SHARED_INVARIANT:
        return "shared_invariant"
    if segment_type in SHARED_APPENDABLE and not mutable:
        return "shared_appendable"
    if segment_type in SHARED_APPENDABLE:
        return "shared_appendable"
    if segment_type in BRANCH_DELTA:
        return "branch_delta"
    return "private_unreusable"


def _domain_id(instance_id: str, invariant_hashes: list[str]) -> str:
    payload = instance_id + "|" + "|".join(sorted(h for h in invariant_hashes if h))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _safe_selected(cls: str, fanout: int, resident_score: float, threshold: float) -> bool:
    if cls == "shared_invariant":
        return fanout >= 2 and resident_score >= threshold
    if cls == "shared_appendable":
        return fanout >= 2 and resident_score >= threshold
    return False


def analyze_events(
    events: list[Event],
    latency_model: LatencyModel | None = None,
    threshold: float = 1.0e-13,
    cfg: dict[str, Any] | None = None,
) -> tuple[list[CDFSegment], dict[str, CDFSegment]]:
    lm = latency_model or LatencyModel()
    bpt = kv_bytes_per_token(cfg)
    seg_defs: dict[tuple[str, str], ContextSegment] = {}
    consumers: dict[tuple[str, str], list[Event]] = defaultdict(list)
    instance_invariant_hashes: dict[str, set[str]] = defaultdict(set)

    for ev in events:
        if ev.node_type != "llm":
            continue
        defs = {seg.segment_id: seg for seg in ev.context_segment_defs}
        for seg in ev.context_segment_defs:
            key = (ev.instance_id, seg.segment_id)
            seg_defs.setdefault(key, seg)
            if seg.segment_type in SHARED_INVARIANT:
                instance_invariant_hashes[ev.instance_id].add(seg.token_hash)
        for ref in ev.context_segments:
            key = (ev.instance_id, ref.segment_id)
            if key not in seg_defs and ref.segment_id in defs:
                seg_defs[key] = defs[ref.segment_id]
            consumers[key].append(ev)

    rows: list[CDFSegment] = []
    selected: dict[str, CDFSegment] = {}
    for key, seg in sorted(seg_defs.items()):
        instance_id, segment_id = key
        evs = consumers.get(key, [])
        branch_ids = sorted({ev.branch_id for ev in evs})
        llm_ids = sorted({ev.node_id for ev in evs})
        fanout = len(set(llm_ids))
        access_count = len(evs)
        cls = segment_class(seg.segment_type, seg.mutable)
        reuse_count = max(0, access_count - 1)
        prefill = lm.predict_prefill(seg.length)
        kvb = seg.length * bpt
        utility = fanout * reuse_count * prefill
        estimated_noc = 0.05 * kvb
        eviction_risk = 0.10 * kvb if cls == "shared_appendable" else 0.02 * kvb
        cost = kvb + estimated_noc + eviction_risk
        score = utility / max(cost, 1e-12)
        domain = _domain_id(instance_id, list(instance_invariant_hashes.get(instance_id, set())))
        is_selected = _safe_selected(cls, fanout, score, threshold)
        row = CDFSegment(
            segment_id=segment_id,
            segment_type=seg.segment_type,
            segment_class=cls,
            instance_id=instance_id,
            length=seg.length,
            token_hash=seg.token_hash,
            start_pos=seg.start_pos,
            exact_prefix_reusable=seg.exact_prefix_reusable,
            fanout=fanout,
            access_count=access_count,
            consumer_branch_ids=branch_ids,
            consumer_llm_node_ids=llm_ids,
            prefill_cost=prefill,
            kv_bytes=kvb,
            utility=utility,
            cost=cost,
            resident_score=score,
            cdf_selected=is_selected,
            context_domain_id=domain,
        )
        rows.append(row)
        if is_selected:
            selected[segment_id] = row
    return rows, selected


def selected_segment_ids(events: list[Event], latency_model: LatencyModel | None = None, threshold: float = 1.0e-13) -> set[str]:
    _, selected = analyze_events(events, latency_model=latency_model, threshold=threshold)
    return set(selected)


def _aggregate_rows(segments: list[CDFSegment], run_id: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for seg in segments:
        key = (seg.instance_id, seg.segment_type)
        row = grouped.setdefault(
            key,
            {
                "run_id": run_id,
                "instance_id": seg.instance_id,
                "segment_type": seg.segment_type,
                "original_repeated_prefill_tokens": 0,
                "original_exact_prefix_reusable_tokens": 0,
                "cdf_exact_prefix_reusable_tokens": 0,
                "cdf_added_reusable_tokens": 0,
                "reusable_ratio_before": 0.0,
                "reusable_ratio_after": 0.0,
                "estimated_prefill_time_before": 0.0,
                "estimated_prefill_time_after": 0.0,
                "estimated_prefill_time_saved": 0.0,
                "cdf_selected_segments": 0,
                "cdf_kv_bytes": 0,
                "cdf_mode": "replay_potential",
            },
        )
        repeated = max(0, seg.access_count - 1) * seg.length
        original_exact = repeated if seg.exact_prefix_reusable and seg.fanout >= 2 else 0
        cdf_exact = original_exact
        if seg.cdf_selected:
            cdf_exact = max(cdf_exact, repeated)
            row["cdf_selected_segments"] += 1
            row["cdf_kv_bytes"] += seg.kv_bytes
        added = max(0, cdf_exact - original_exact)
        before_tokens = max(0, repeated - original_exact)
        after_tokens = max(0, repeated - cdf_exact)
        row["original_repeated_prefill_tokens"] += repeated
        row["original_exact_prefix_reusable_tokens"] += original_exact
        row["cdf_exact_prefix_reusable_tokens"] += cdf_exact
        row["cdf_added_reusable_tokens"] += added
        row["estimated_prefill_time_before"] += seg.prefill_cost * (before_tokens / max(1, seg.length))
        row["estimated_prefill_time_after"] += seg.prefill_cost * (after_tokens / max(1, seg.length))
        row["estimated_prefill_time_saved"] += seg.prefill_cost * (added / max(1, seg.length))
    for row in grouped.values():
        repeated = float(row["original_repeated_prefill_tokens"])
        row["reusable_ratio_before"] = float(row["original_exact_prefix_reusable_tokens"]) / repeated if repeated else 0.0
        row["reusable_ratio_after"] = float(row["cdf_exact_prefix_reusable_tokens"]) / repeated if repeated else 0.0
    return list(grouped.values())


def compare_context_reuse(
    trace_dir: str | Path = "data/traces/mini_swe_lite10_r4_timed",
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    run_id: str = "pr4_algo",
    out_csv: str | Path = "data/results/cdf_context_reuse_comparison.csv",
    detail_json: str | Path = "data/results/cdf_context_domains_pr4_algo.json",
    plot_out: str | Path = "data/plots/cdf_context_reuse_pr4_algo.pdf",
) -> list[dict[str, Any]]:
    traces = load_trace_dir(trace_dir)
    events = [ev for trace in traces for ev in trace.events]
    lm = LatencyModel.load(model_json)
    segments, _ = analyze_events(events, lm)
    rows = _aggregate_rows(segments, run_id)
    write_csv(out_csv, rows)
    domains: dict[str, dict[str, Any]] = {}
    for seg in segments:
        dom = domains.setdefault(
            seg.context_domain_id,
            {
                "context_domain_id": seg.context_domain_id,
                "instances": set(),
                "invariant_segments": [],
                "appendable_segments": [],
                "consumer_branch_ids": set(),
                "consumer_llm_node_ids": set(),
                "total_tokens": 0,
                "kv_bytes": 0,
                "fanout": 0,
                "reuse_count": 0,
            },
        )
        dom["instances"].add(seg.instance_id)
        if seg.segment_class == "shared_invariant":
            dom["invariant_segments"].append(seg.segment_id)
        elif seg.segment_class == "shared_appendable":
            dom["appendable_segments"].append(seg.segment_id)
        dom["consumer_branch_ids"].update(seg.consumer_branch_ids)
        dom["consumer_llm_node_ids"].update(seg.consumer_llm_node_ids)
        dom["total_tokens"] += seg.length
        dom["kv_bytes"] += seg.kv_bytes
        dom["fanout"] = max(dom["fanout"], seg.fanout)
        dom["reuse_count"] += max(0, seg.access_count - 1)
    serializable = []
    for dom in domains.values():
        serializable.append(
            {
                **dom,
                "instances": sorted(dom["instances"]),
                "consumer_branch_ids": sorted(dom["consumer_branch_ids"]),
                "consumer_llm_node_ids": sorted(dom["consumer_llm_node_ids"]),
            }
        )
    write_json(detail_json, {"cdf_mode": "replay_potential", "domains": serializable})
    plot_cdf_reuse(rows, plot_out)
    return rows


def plot_cdf_reuse(rows: list[dict[str, Any]], out: str | Path) -> None:
    ensure_dir(Path(out).parent)
    agg: dict[str, dict[str, float]] = defaultdict(lambda: {"original": 0.0, "added": 0.0, "before": 0.0, "after": 0.0})
    for row in rows:
        typ = str(row.get("segment_type", "unknown"))
        agg[typ]["original"] += float(row.get("original_exact_prefix_reusable_tokens", 0) or 0)
        agg[typ]["added"] += float(row.get("cdf_added_reusable_tokens", 0) or 0)
        agg[typ]["before_num"] = agg[typ].get("before_num", 0.0) + float(row.get("original_exact_prefix_reusable_tokens", 0) or 0)
        agg[typ]["after_num"] = agg[typ].get("after_num", 0.0) + float(row.get("cdf_exact_prefix_reusable_tokens", 0) or 0)
        agg[typ]["den"] = agg[typ].get("den", 0.0) + float(row.get("original_repeated_prefill_tokens", 0) or 0)
    labels = sorted(agg)
    if not labels:
        labels = ["none"]
        original = [0]
        added = [0]
        before = [0]
        after = [0]
    else:
        original = [agg[x]["original"] for x in labels]
        added = [agg[x]["added"] for x in labels]
        before = [agg[x].get("before_num", 0.0) / max(1.0, agg[x].get("den", 0.0)) for x in labels]
        after = [agg[x].get("after_num", 0.0) / max(1.0, agg[x].get("den", 0.0)) for x in labels]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    x = range(len(labels))
    axes[0].bar([i - 0.18 for i in x], original, width=0.36, label="original exact reusable")
    axes[0].bar([i + 0.18 for i in x], added, width=0.36, label="CDF added reusable")
    axes[0].set_xticks(list(x), labels, rotation=25, ha="right")
    axes[0].set_ylabel("tokens")
    axes[0].legend(fontsize=8)
    axes[1].plot(list(x), before, marker="o", label="before")
    axes[1].plot(list(x), after, marker="o", label="after")
    axes[1].set_xticks(list(x), labels, rotation=25, ha="right")
    axes[1].set_ylabel("reusable ratio")
    axes[1].set_ylim(0, max(1.0, max(after or [0]) * 1.1))
    axes[1].legend(fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default="data/traces/mini_swe_lite10_r4_timed")
    ap.add_argument("--model-json", default="data/profiles/h100_latency_model_pr2_v2.json")
    ap.add_argument("--run-id", default="pr4_algo")
    ap.add_argument("--out", default="data/results/cdf_context_reuse_comparison.csv")
    ap.add_argument("--detail-json", default="data/results/cdf_context_domains_pr4_algo.json")
    ap.add_argument("--plot-out", default="data/plots/cdf_context_reuse_pr4_algo.pdf")
    args = ap.parse_args()
    rows = compare_context_reuse(args.trace_dir, args.model_json, args.run_id, args.out, args.detail_json, args.plot_out)
    total_added = sum(int(r.get("cdf_added_reusable_tokens", 0)) for r in rows)
    print(json.dumps({"rows": len(rows), "cdf_added_reusable_tokens": total_added}, indent=2))


if __name__ == "__main__":
    main()
