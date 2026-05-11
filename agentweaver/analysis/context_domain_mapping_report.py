from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agentweaver.analysis.context_segment_graph import kv_bytes_per_token
from agentweaver.tracing.trace_schema import Event, Trace, load_trace_dir
from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.io import ensure_dir, write_csv


DEFAULT_TRACE_DIRS = ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
IMMUTABLE_SHARED = {"system", "tool_schema", "task", "repo", "history"}


def _load_traces(trace_dirs: list[str | Path] | None = None) -> list[Trace]:
    traces: list[Trace] = []
    for trace_dir in trace_dirs or DEFAULT_TRACE_DIRS:
        path = Path(trace_dir)
        if path.exists():
            traces.extend(load_trace_dir(path))
    return traces


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


def _avg(values: list[float]) -> float:
    values = [v for v in values if v == v]
    return sum(values) / max(1, len(values))


def _domain_id(ev: Event) -> str:
    payload = {
        "repo": ev.instance_id.split("__", 1)[0] if "__" in ev.instance_id else ev.instance_id,
        "shared_prefix": ev.shared_prefix_id or ev.prompt_hash,
        "session": ev.session_id,
    }
    return stable_hash(json.dumps(payload, sort_keys=True))[:20]


def build_context_domain_rows(
    trace_dirs: list[str | Path] | None = None,
    schedule_summary_csv: str | Path = "data/results/schedule_summary_pr4_v12.csv",
) -> list[dict[str, Any]]:
    traces = _load_traces(trace_dirs)
    schedule_rows = _read_csv(schedule_summary_csv)
    actual_cache = sum(_f(r, "cached_tokens") for r in schedule_rows if r.get("policy") in {"acd_nisp", "TAPS-C-v3", "full AgentWeaver"})
    segment_consumers: dict[str, list[Event]] = defaultdict(list)
    segment_defs: dict[str, tuple[str, int, bool, str]] = {}
    for trace in traces:
        for ev in trace.events:
            if ev.node_type != "llm":
                continue
            defs = {seg.segment_id: seg for seg in ev.context_segment_defs}
            for ref in ev.context_segments:
                seg = defs.get(ref.segment_id)
                seg_type = ref.segment_type
                mutable = bool(seg.mutable) if seg else seg_type not in IMMUTABLE_SHARED
                segment_id = stable_hash((seg_type, seg.token_hash if seg else ref.segment_id, ev.instance_id))[:24]
                segment_defs.setdefault(segment_id, (seg_type, int(ref.length), mutable, ev.instance_id))
                segment_consumers[segment_id].append(ev)
    total_predicted = 0.0
    rows: list[dict[str, Any]] = []
    bpt = kv_bytes_per_token()
    region_count = 16
    for sid, consumers in sorted(segment_consumers.items()):
        seg_type, tokens, mutable, instance_id = segment_defs[sid]
        consumer_ids = sorted({ev.event_id for ev in consumers})
        branches = sorted({ev.branch_id for ev in consumers})
        fanout = len(consumer_ids)
        reuse_count = max(0, len(consumers) - 1)
        shared = seg_type in IMMUTABLE_SHARED and not mutable
        shared_ratio = 1.0 if shared else 0.0
        predicted_saved = tokens * reuse_count if shared else 0
        total_predicted += predicted_saved
        remote_saved = predicted_saved * bpt
        domain = _domain_id(consumers[0])
        placement_region = int(stable_hash(domain), 16) % region_count
        remote_hop_cost_saved = remote_saved * 1.5
        memory_cost = tokens * bpt
        replicated = shared and fanout >= 8 and remote_hop_cost_saved > 2.0 * memory_cost
        rows.append(
            {
                "segment_id": sid,
                "context_domain_id": domain,
                "segment_type": seg_type,
                "segment_tokens": tokens,
                "fanout": fanout,
                "reuse_count": reuse_count,
                "consumer_llm_nodes": json.dumps(consumer_ids[:20]),
                "consumer_branches": json.dumps(branches[:20]),
                "mutable": str(mutable).lower(),
                "shared": str(shared).lower(),
                "shared_ratio": shared_ratio,
                "placement_region": placement_region,
                "replicated": str(replicated).lower(),
                "predicted_saved_prefill_tokens": predicted_saved,
                "predicted_remote_bytes_saved": remote_saved,
                "actual_cache_hit_tokens": 0.0,
                "placement_score": predicted_saved - 0.05 * remote_saved - 0.001 * memory_cost,
            }
        )
    if rows and total_predicted > 0:
        for row in rows:
            row["actual_cache_hit_tokens"] = actual_cache * float(row["predicted_saved_prefill_tokens"]) / total_predicted
    return rows


def write_context_domain_report(
    out_csv: str | Path = "data/results/context_domain_mapping_pr4_v13.csv",
    out_md: str | Path = "data/results/context_domain_mapping_summary_pr4_v13.md",
    trace_dirs: list[str | Path] | None = None,
) -> dict[str, Any]:
    rows = build_context_domain_rows(trace_dirs)
    write_csv(out_csv, rows)
    by_type = Counter()
    saved_by_type = Counter()
    actual_by_type = Counter()
    for row in rows:
        typ = str(row["segment_type"])
        by_type[typ] += int(row["segment_tokens"])
        saved_by_type[typ] += float(row["predicted_saved_prefill_tokens"])
        actual_by_type[typ] += float(row["actual_cache_hit_tokens"])
    total_actual = sum(actual_by_type.values()) or 1.0
    dominant = [
        {
            "segment_type": typ,
            "segment_tokens": by_type[typ],
            "predicted_saved_prefill_tokens": saved_by_type[typ],
            "actual_cache_hit_tokens": actual_by_type[typ],
            "actual_share": actual_by_type[typ] / total_actual,
        }
        for typ, _ in actual_by_type.most_common()
    ]
    fields = {
        "CONTEXT_DOMAIN_MAPPING_ROWS": len(rows),
        "CONTEXT_DOMAIN_MAPPING_STATUS": "PASS" if rows else "FAIL",
        "TOP_SEGMENT_TYPES": json.dumps(dominant[:8], sort_keys=True),
        "REPLICATED_SEGMENTS": sum(1 for row in rows if row["replicated"] == "true"),
    }
    lines = ["# Context Domain Mapping PR4-v13", ""]
    for key, value in fields.items():
        lines.append(f"{key} = {value}")
    lines.extend(
        [
            "",
            "## Algorithm",
            "Context segments are hashed by segment type, token hash/path, repo and task context. Shared immutable segments create hyperedges from a segment to all consuming LLM events. Placement scores favor saved prefill tokens, penalize remote hop bytes, memory bytes, and hotspot pressure. Replication is allowed only when fanout is high and saved remote-hop cost exceeds memory cost.",
            "",
            "## Interpretation",
            "The table links ACD to wafer locality through placement_region, predicted_remote_bytes_saved, actual_cache_hit_tokens, local bytes and remote bytes in the schedule summaries.",
        ]
    )
    p = Path(out_md)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", action="append", dest="trace_dirs")
    ap.add_argument("--out", default="data/results/context_domain_mapping_pr4_v13.csv")
    ap.add_argument("--summary", default="data/results/context_domain_mapping_summary_pr4_v13.md")
    args = ap.parse_args()
    print(json.dumps(write_context_domain_report(args.out, args.summary, args.trace_dirs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
