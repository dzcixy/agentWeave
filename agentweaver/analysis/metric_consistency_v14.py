from __future__ import annotations

import argparse
import csv
import json
import glob
from pathlib import Path
from typing import Any


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


def _schedule_sum(path: str | Path, key: str) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    total = 0.0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += _f(json.loads(line), key)
    return total


def _replicate_schedule_paths(path: str | Path) -> list[Path]:
    p = Path(path)
    if not p.exists():
        return []
    text = str(p)
    if "_r" in p.stem:
        prefix = text.rsplit("_r", 1)[0]
        matches = sorted(Path(m) for m in glob.glob(prefix + "_r*.jsonl"))
        return matches or [p]
    return [p]


def _schedule_sum_many(paths: list[Path], key: str) -> float:
    return sum(_schedule_sum(path, key) for path in paths)


def write_metric_dictionary(out_md: str | Path = "data/results/metric_dictionary_pr4_v14.md") -> None:
    text = """# Metric Dictionary PR4-v14

prefill_compute_tokens = uncached input tokens charged to prefill compute after ACD/NISP cache hits.
resume_prefill_tokens = tokens recomputed on LLM resume; equals uncached prefill work in replay.
cache_hit_tokens = shared_context_hit_tokens + private_suffix_hit_tokens.
shared_context_hit_tokens = immutable shared segment tokens served by ACD residency.
private_suffix_hit_tokens = branch-private suffix tokens preserved by NISP parking.
local_context_bytes = cache/resident bytes read locally on wafer or HBM.
remote_context_bytes = payload context bytes that must move remotely before hop weighting.
remote_kv_bytes = charged NoC bytes used by the simulator, including hop/placement effects.
schedule_remote_kv_bytes = sum of remote_kv_bytes in the replay schedule JSONL(s) at the row's aggregation grain.
astra_policy_remote_bytes = communication bytes emitted by ASTRA exporter; must match the exact schedule JSONL exported.
model_side_latency = prefill_latency + decode_latency + local_memory_latency + noC_latency + queueing/state-prefetch latency.
noC_latency = remote transfer latency plus hop and contention penalty.
tool_latency = observed trace tool latency, reduced only for explicitly enabled safe STP.
end_to_end_jct = model_side_latency + tool_latency on the replayed critical path.
"""
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text(text, encoding="utf-8")


def run_consistency(
    replay_csv: str | Path = "data/results/agentweaver_v14_mode_replay.csv",
    schedule_summary_csv: str | Path = "data/results/schedule_summary_pr4_v14.csv",
    astra_csv: str | Path = "data/results/astra_export_v14_rows.csv",
    out_md: str | Path = "data/results/metric_consistency_pr4_v14.md",
) -> dict[str, Any]:
    replay = _read_csv(replay_csv)
    summary = _read_csv(schedule_summary_csv)
    astra = _read_csv(astra_csv)
    failures: list[str] = []
    for row in replay:
        if abs(_f(row, "remote_kv_bytes") - _f(row, "schedule_remote_kv_bytes")) > max(1.0, _f(row, "remote_kv_bytes")) * 1e-9:
            failures.append(f"remote_kv_mismatch:{row.get('mode')}:{row.get('config_id')}:{row.get('replicate_id')}")
        if abs(_f(row, "cache_hit_tokens") - (_f(row, "shared_context_hit_tokens") + _f(row, "private_suffix_hit_tokens"))) > 1e-6:
            failures.append(f"cache_class_overlap:{row.get('mode')}:{row.get('config_id')}:{row.get('replicate_id')}")
        if _f(row, "prefill_compute_tokens") + _f(row, "cache_hit_tokens") + 1e-6 < _f(row, "input_tokens"):
            failures.append(f"token_accounting_underflow:{row.get('mode')}:{row.get('config_id')}:{row.get('replicate_id')}")
    for row in summary:
        sched = row.get("schedule_jsonl", "")
        if not sched:
            continue
        sched_remote = _schedule_sum_many(_replicate_schedule_paths(sched), "remote_kv_bytes")
        if abs(sched_remote - _f(row, "schedule_remote_kv_bytes")) > max(1.0, sched_remote) * 1e-6:
            failures.append(f"schedule_remote_sum:{row.get('mode')}:{row.get('config_id')}")
    for ar in astra:
        sched = ar.get("schedule_jsonl", "")
        if not sched:
            continue
        sched_remote = _schedule_sum(sched, "remote_kv_bytes")
        rel = abs(_f(ar, "astra_policy_remote_bytes") - sched_remote) / max(1.0, sched_remote)
        if rel > 0.01:
            failures.append(f"astra_remote_mismatch:{ar.get('mode', ar.get('policy', ''))}:{ar.get('config_id')}:{rel:.6f}")
    fields = {
        "METRIC_CONSISTENCY_PASS": str(not failures).lower(),
        "REPLAY_ROWS": len(replay),
        "SCHEDULE_SUMMARY_ROWS": len(summary),
        "ASTRA_ROWS": len(astra),
        "FAILURE_COUNT": len(failures),
        "FAILURES": json.dumps(failures[:50]),
    }
    lines = ["# Metric Consistency PR4-v14", ""]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_metric_dictionary()
    return fields


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default="data/results/agentweaver_v14_mode_replay.csv")
    ap.add_argument("--schedule-summary", default="data/results/schedule_summary_pr4_v14.csv")
    ap.add_argument("--astra", default="data/results/astra_export_v14_rows.csv")
    ap.add_argument("--out", default="data/results/metric_consistency_pr4_v14.md")
    args = ap.parse_args()
    print(json.dumps(run_consistency(args.replay, args.schedule_summary, args.astra, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
