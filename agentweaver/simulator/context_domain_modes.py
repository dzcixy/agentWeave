from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SHARED_SEGMENT_TYPES = {"system", "tool_schema", "task", "repo", "history"}
PRIVATE_SEGMENT_TYPES = {"observation", "branch_suffix", "test_log", "patch", "scratchpad", "unknown"}


def read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_schedule_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def f(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return default
    try:
        value = row.get(key)
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def avg(values: list[float]) -> float:
    values = [v for v in values if v == v]
    return sum(values) / max(1, len(values))


def aggregate_schedule(summary_csv: str | Path = "data/results/schedule_summary_pr4_v12.csv") -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(summary_csv):
        groups[str(row.get("policy", ""))].append(row)
    out: dict[str, dict[str, float]] = {}
    for policy, rows in groups.items():
        out[policy] = {
            "cached_tokens": avg([f(r, "cached_tokens") for r in rows]),
            "recompute_tokens": avg([f(r, "recompute_tokens") for r in rows]),
            "local_context_bytes": avg([f(r, "local_context_bytes") for r in rows]),
            "remote_context_bytes": avg([f(r, "remote_context_bytes") for r in rows]),
            "remote_kv_bytes": avg([f(r, "schedule_remote_kv_bytes") for r in rows]),
            "memory_occupancy": avg([f(r, "memory_occupancy_after", f(r, "memory_occupancy", 0.0)) for r in rows]),
            "tool_latency_hidden": avg([f(r, "tool_latency_hidden") for r in rows]),
        }
    return out


def estimate_acd_shared_metrics(summary_csv: str | Path = "data/results/schedule_summary_pr4_v12.csv") -> dict[str, float]:
    schedules = read_csv(summary_csv)
    acd_rows = [r for r in schedules if r.get("policy") == "acd_nisp"]
    taps_rows = [r for r in schedules if r.get("policy") == "TAPS-C-v3"]
    cached = avg([f(r, "cached_tokens") for r in acd_rows])
    local = avg([f(r, "local_context_bytes") for r in acd_rows])
    remote = avg([f(r, "remote_context_bytes") for r in acd_rows])
    taps_remote = avg([f(r, "remote_context_bytes") for r in taps_rows])
    raw_remote = local + remote
    remote_reduction = (raw_remote - remote) / max(1.0, raw_remote)
    return {
        "shared_context_hit_tokens": cached,
        "predicted_saved_prefill_tokens": cached,
        "local_context_bytes": local,
        "remote_context_bytes": remote,
        "remote_kv_bytes": avg([f(r, "schedule_remote_kv_bytes") for r in acd_rows]),
        "remote_reduction_vs_raw_context": remote_reduction,
        "remote_reduction_vs_taps_schedule": (taps_remote - remote) / max(1.0, taps_remote) if taps_remote else 0.0,
    }


def summarize_context_domains(schedule_summary_csv: str | Path = "data/results/schedule_summary_pr4_v12.csv") -> dict[str, Any]:
    rows = read_csv(schedule_summary_csv)
    domain_counts: Counter[str] = Counter()
    local_by_domain: Counter[str] = Counter()
    remote_by_domain: Counter[str] = Counter()
    for summary in rows:
        if summary.get("policy") not in {"acd_nisp", "TAPS-C-v3", "full AgentWeaver"}:
            continue
        for event in read_schedule_jsonl(summary.get("schedule_jsonl", "")):
            domain = str(event.get("context_domain_id", "unknown"))
            domain_counts[domain] += 1
            local_by_domain[domain] += int(event.get("local_context_bytes", 0) or 0)
            remote_by_domain[domain] += int(event.get("remote_context_bytes", 0) or 0)
    total = sum(domain_counts.values()) or 1
    return {
        "context_domain_count": len(domain_counts),
        "top_context_domains": [
            {
                "context_domain_id": domain,
                "event_share": count / total,
                "local_context_bytes": local_by_domain[domain],
                "remote_context_bytes": remote_by_domain[domain],
            }
            for domain, count in domain_counts.most_common(10)
        ],
    }
