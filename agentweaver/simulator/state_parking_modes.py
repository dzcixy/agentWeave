from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


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


def state_parking_rows(schedule_summary_csv: str | Path = "data/results/schedule_summary_pr4_v12.csv") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in read_csv(schedule_summary_csv):
        if summary.get("policy") != "acd_nisp":
            continue
        for event in read_schedule_jsonl(summary.get("schedule_jsonl", "")):
            residency = str(event.get("state_residency", "NONE"))
            parked = int(event.get("parked_state_bytes", 0) or 0)
            recompute = int(event.get("recompute_tokens", 0) or 0)
            cached = int(event.get("cached_tokens", 0) or 0)
            rows.append(
                {
                    "run_id": event.get("run_id", ""),
                    "config_id": event.get("config_id", ""),
                    "branch_id": event.get("branch_id", ""),
                    "event_id": event.get("event_id", ""),
                    "tool_type": "unknown",
                    "tool_latency": 0.0,
                    "predicted_tool_latency": 0.0,
                    "state_residency": residency,
                    "shared_prefix_bytes": int(event.get("local_context_bytes", 0) or 0),
                    "private_suffix_bytes": parked,
                    "observation_delta_bytes": max(0, int(event.get("remote_context_bytes", 0) or 0) - parked),
                    "parked_state_bytes": parked,
                    "recompute_tokens_if_evicted": recompute + cached,
                    "actual_reuse_after_tool": cached > 0 or parked > 0,
                    "memory_pressure": float(event.get("memory_occupancy_after", 0) or 0) / max(1.0, float(event.get("memory_occupancy_after", 0) or 0) + float(event.get("remote_context_bytes", 0) or 0)),
                    "eviction_reason": "none" if residency in {"HOT", "WARM"} else "not_parked_or_cold",
                }
            )
    return rows


def estimate_nisp_private_metrics(schedule_summary_csv: str | Path = "data/results/schedule_summary_pr4_v12.csv") -> dict[str, float]:
    rows = state_parking_rows(schedule_summary_csv)
    counts = Counter(str(r["state_residency"]) for r in rows)
    parked_bytes = sum(int(r["parked_state_bytes"]) for r in rows)
    recompute_if_evicted = sum(int(r["recompute_tokens_if_evicted"]) for r in rows if r["state_residency"] in {"HOT", "WARM"})
    # Private suffix savings are attributed only to parked state. Shared ACD cache hits
    # are intentionally excluded to avoid double-counting.
    private_suffix_hit_tokens = parked_bytes / 4096.0
    return {
        "hot_stalls": counts.get("HOT", 0),
        "warm_stalls": counts.get("WARM", 0),
        "cold_stalls": counts.get("COLD", 0),
        "none_stalls": counts.get("NONE", 0),
        "parked_state_bytes": parked_bytes,
        "private_suffix_hit_tokens": private_suffix_hit_tokens,
        "resume_prefill_tokens_saved": private_suffix_hit_tokens,
        "recompute_tokens_if_evicted": recompute_if_evicted,
    }
