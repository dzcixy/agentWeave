from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentweaver.tracing.trace_schema import Trace, load_trace_dir, validate_trace
from agentweaver.utils.io import ensure_dir, write_csv


def _ordered(events):
    return sorted(events, key=lambda e: (e.branch_id, e.step_id, e.timestamp_start, e.node_id))


def _has_pair(trace: Trace, first: str, second: str) -> bool:
    for branch in sorted({e.branch_id for e in trace.events}):
        events = [e for e in _ordered(trace.events) if e.branch_id == branch]
        for a, b in zip(events, events[1:]):
            if a.node_type == first and b.node_type == second:
                return True
    return False


def _row(name: str, traces: list[Trace]) -> dict[str, Any]:
    events = [e for tr in traces for e in tr.events]
    errors = [err for tr in traces for err in validate_trace(tr)]
    llm = [e for e in events if e.node_type == "llm"]
    tool = [e for e in events if e.node_type == "tool"]
    verifier = [e for e in events if e.node_type == "verifier"]
    missing_timestamps = sum(
        1
        for e in events
        if getattr(e, "timing_missing", False)
        or ((e.node_type in {"llm", "tool", "verifier"}) and not (e.timestamp_start and e.timestamp_end))
    )
    missing_token_counts = sum(1 for e in llm if e.input_tokens <= 0 or e.output_tokens <= 0)
    missing_context_segments = sum(1 for e in llm if not e.context_segments)
    return {
        "trace": name,
        "num_sessions": len({e.session_id for e in events}),
        "num_instances": len({e.instance_id for e in events}),
        "num_llm_events": len(llm),
        "num_tool_events": len(tool),
        "num_verifier_events": len(verifier),
        "missing_timestamps": missing_timestamps,
        "missing_token_counts": missing_token_counts,
        "missing_context_segments": missing_context_segments,
        "dag_valid": str(not errors).lower(),
        "validation_errors": "; ".join(errors[:20]),
        "branch_count": len({e.branch_id for e in events if e.branch_id != "root"}),
        "has_tool_after_llm": str(any(_has_pair(tr, "llm", "tool") for tr in traces)).lower(),
        "has_llm_after_tool": str(any(_has_pair(tr, "tool", "llm") for tr in traces)).lower(),
    }


def validate_path(trace: str | Path | None = None, trace_dir: str | Path | None = None) -> list[dict[str, Any]]:
    if trace_dir:
        traces = load_trace_dir(trace_dir)
        rows = [_row(Path(tr.metadata.get("source", f"trace_{i}")).name, [tr]) for i, tr in enumerate(traces)]
        rows.append(_row("AGGREGATE", traces))
        return rows
    if trace:
        tr = Trace.from_jsonl(trace)
        return [_row(Path(trace).name, [tr])]
    raise ValueError("expected --trace or --trace-dir")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace")
    ap.add_argument("--trace-dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = validate_path(args.trace, args.trace_dir)
    if args.out:
        ensure_dir(Path(args.out).parent)
        write_csv(args.out, rows)
    else:
        print(json.dumps(rows, indent=2))
    aggregate = rows[-1] if rows else {}
    if str(aggregate.get("dag_valid", "false")).lower() != "true":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
