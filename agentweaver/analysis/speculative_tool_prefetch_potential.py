from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agentweaver.tracing.trace_schema import Event, Trace, load_trace_dir
from agentweaver.utils.io import ensure_dir, write_csv


SAFE_CLASSES = {"grep_search", "file_read", "list_find", "python_inspect", "shell_readonly"}


def _load_traces(trace_dirs: list[str | Path]) -> list[Trace]:
    traces: list[Trace] = []
    for trace_dir in trace_dirs:
        p = Path(trace_dir)
        if p.exists():
            traces.extend(load_trace_dir(p))
    return traces


def command_class(command: str | None, tool_type: str | None = None) -> str:
    cmd = (command or "").strip().lower()
    if not cmd:
        return tool_type or "unknown"
    if any(x in cmd for x in ["pytest", "tox", "unittest", "manage.py test"]):
        return "test"
    if cmd.startswith(("rg ", "grep ")):
        return "grep_search"
    if cmd.startswith(("cat ", "sed ", "head ", "tail ")):
        return "file_read"
    if cmd.startswith(("ls", "find ", "pwd", "tree ")):
        return "list_find"
    if "python" in cmd and not any(x in cmd for x in [">", "write", "open(", "pip install"]):
        return "python_inspect"
    if any(x in cmd for x in ["apply_patch", ">>", ">", "mv ", "cp ", "rm ", "touch ", "git apply"]):
        return "file_write"
    return "shell_other"


def _tool_latency(ev: Event) -> float:
    return float(ev.tool_latency if ev.tool_latency is not None else ev.latency or 0.0)


def _events_by_branch(trace: Trace) -> dict[str, list[Event]]:
    out: dict[str, list[Event]] = defaultdict(list)
    for ev in sorted(trace.events, key=lambda e: (e.branch_id, e.step_id, e.timestamp_start or 0.0)):
        out[ev.branch_id].append(ev)
    return out


def analyze_prefetch_potential(
    trace_dirs: list[str | Path] | None = None,
    out_csv: str | Path = "data/results/speculative_tool_prefetch_potential_pr4_v9.csv",
    out_md: str | Path = "data/results/speculative_tool_prefetch_potential_pr4_v9.md",
) -> dict[str, Any]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    traces = _load_traces(trace_dirs)
    examples: list[dict[str, Any]] = []
    train_counts: Counter[tuple[str, str]] = Counter()
    prior_counts: Counter[str] = Counter()
    records: list[tuple[str, str, float, bool]] = []
    for trace in traces:
        repo = str(trace.metadata.get("repo", "") or trace.metadata.get("instance_id", "")).split("__")[0]
        for events in _events_by_branch(trace).values():
            prev_class = "START"
            for idx, ev in enumerate(events):
                if ev.node_type != "tool":
                    continue
                cls = command_class(ev.command, ev.tool_type)
                train_counts[(prev_class, cls)] += 1
                prior_counts[cls] += 1
                records.append((prev_class, cls, _tool_latency(ev), cls in SAFE_CLASSES))
                prev_class = cls
                if len(examples) < 20:
                    examples.append({"repo": repo, "prev_command_class": records[-1][0], "next_command_class": cls, "safe": cls in SAFE_CLASSES})

    total = len(records)
    top1 = 0
    top3 = 0
    safe_correct_latency = 0.0
    safe_total = 0
    global_top = [c for c, _ in prior_counts.most_common(3)]
    for prev, actual, latency, safe in records:
        candidates = [c for (p, c), _ in train_counts.most_common() if p == prev][:3]
        if not candidates:
            candidates = global_top
        if candidates and candidates[0] == actual:
            top1 += 1
            if safe:
                safe_correct_latency += latency
        if actual in candidates:
            top3 += 1
        if safe:
            safe_total += 1
    safe_coverage = safe_total / max(1, total)
    rows = examples + [
        {
            "repo": "aggregate",
            "prev_command_class": "",
            "next_command_class": "",
            "safe": "",
            "total_tool_events": total,
            "safe_tool_coverage": safe_coverage,
            "next_tool_top1_acc": top1 / max(1, total),
            "next_tool_top3_acc": top3 / max(1, total),
            "potential_latency_saved": safe_correct_latency,
        }
    ]
    write_csv(out_csv, rows)
    fields = {
        "SPECULATIVE_TOOL_PREFETCH_ANALYSIS": "PASS" if total else "WARNING",
        "SAFE_TOOL_COVERAGE": f"{safe_coverage:.6f}",
        "NEXT_TOOL_TOP1_ACC": f"{top1 / max(1, total):.6f}",
        "NEXT_TOOL_TOP3_ACC": f"{top3 / max(1, total):.6f}",
        "POTENTIAL_LATENCY_SAVED": f"{safe_correct_latency:.6f}",
    }
    lines = [
        "# Speculative Tool Prefetch Potential PR4-v9",
        "",
        "This is a potential analysis only. It considers safe read-only/idempotent command classes and is not used as a main AgentWeaver result.",
        "",
    ]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    p = Path(out_md)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", action="append", dest="trace_dirs")
    args = ap.parse_args()
    print(analyze_prefetch_potential(args.trace_dirs))


if __name__ == "__main__":
    main()
