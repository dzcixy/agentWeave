from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from agentweaver.utils.io import ensure_dir

NodeType = Literal["llm", "tool", "retrieval", "rerank", "message", "state", "fork", "join", "verifier"]
SegmentType = Literal[
    "system",
    "tool_schema",
    "task",
    "repo",
    "history",
    "observation",
    "scratchpad",
    "branch_suffix",
    "patch",
    "test_log",
    "unknown",
]
Privacy = Literal["public", "session_private", "branch_private"]
VerifierResult = Literal["pass", "fail", "unknown"]


@dataclass
class ContextSegment:
    segment_id: str
    segment_type: SegmentType
    token_hash: str
    start_pos: int
    length: int
    model: str
    tokenizer: str
    mutable: bool = False
    privacy: Privacy = "session_private"
    exact_prefix_reusable: bool = True


@dataclass
class ContextSegmentRef:
    segment_id: str
    segment_type: SegmentType
    start_pos: int
    length: int
    kv_bytes: Optional[int] = None


@dataclass
class Event:
    event_id: str
    session_id: str
    instance_id: str
    branch_id: str
    parent_branch_id: Optional[str]
    step_id: int
    node_id: str
    node_type: NodeType
    role: Optional[str] = None
    model: Optional[str] = None
    timestamp_ready: float = 0.0
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    latency: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    context_length: int = 0
    prompt_hash: str = ""
    shared_prefix_id: Optional[str] = None
    context_segments: list[ContextSegmentRef] = field(default_factory=list)
    context_segment_defs: list[ContextSegment] = field(default_factory=list)
    prefill_latency: Optional[float] = None
    decode_latency: Optional[float] = None
    queue_latency: Optional[float] = None
    ttft: Optional[float] = None
    tpot: Optional[float] = None
    kv_cache_hit_tokens: Optional[int] = None
    kv_cache_miss_tokens: Optional[int] = None
    tool_type: Optional[str] = None
    command: Optional[str] = None
    tool_latency: Optional[float] = None
    observation_tokens: Optional[int] = None
    exit_code: Optional[int] = None
    verifier_result: Optional[VerifierResult] = None
    patch_id: Optional[str] = None
    patch_hash: Optional[str] = None
    patch_snapshot_available: bool = False
    modified_files_count: int = 0
    untracked_files_count: int = 0
    git_diff_stat_bytes: int = 0
    git_diff_name_count: int = 0
    patch_hash_prefix: Optional[str] = None
    file_modification_seen: bool = False
    is_first_success: bool = False
    success: Optional[bool] = None
    timing_missing: bool = False
    rollout_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.timestamp_end and self.timestamp_start and not self.latency:
            self.latency = max(0.0, self.timestamp_end - self.timestamp_start)
        self.context_segments = [
            x if isinstance(x, ContextSegmentRef) else ContextSegmentRef(**x) for x in self.context_segments
        ]
        self.context_segment_defs = [
            x if isinstance(x, ContextSegment) else ContextSegment(**x) for x in self.context_segment_defs
        ]


@dataclass
class Trace:
    metadata: dict[str, Any]
    events: list[Event]

    def to_jsonl(self, path: str | Path) -> None:
        p = Path(path)
        ensure_dir(p.parent)
        meta = {"record_type": "metadata", **self.metadata}
        with p.open("w", encoding="utf-8") as f:
            f.write(json.dumps(meta, sort_keys=True) + "\n")
            for ev in self.events:
                row = asdict(ev)
                row["record_type"] = "event"
                f.write(json.dumps(row, sort_keys=True) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Trace":
        metadata: dict[str, Any] = {}
        events: list[Event] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                typ = row.pop("record_type", "event")
                if typ == "metadata":
                    metadata = row
                else:
                    events.append(Event(**row))
        if "source" not in metadata:
            metadata["source"] = str(path)
        return cls(metadata=metadata, events=events)


def validate_trace(trace: Trace) -> list[str]:
    errors: list[str] = []
    last_by_branch: dict[str, tuple[int, float]] = {}
    ids: set[str] = set()
    for ev in sorted(trace.events, key=lambda e: (e.branch_id, e.step_id, e.timestamp_start)):
        if ev.event_id in ids:
            errors.append(f"duplicate event_id {ev.event_id}")
        ids.add(ev.event_id)
        if ev.timing_missing:
            prev = last_by_branch.get(ev.branch_id)
            last_by_branch[ev.branch_id] = (ev.step_id, prev[1] if prev else ev.timestamp_end)
            continue
        if ev.timestamp_start < ev.timestamp_ready:
            errors.append(f"{ev.event_id}: start before ready")
        if ev.timestamp_end < ev.timestamp_start:
            errors.append(f"{ev.event_id}: end before start")
        if abs((ev.timestamp_end - ev.timestamp_start) - ev.latency) > 1e-3 and ev.timestamp_end:
            errors.append(f"{ev.event_id}: latency mismatch")
        prev = last_by_branch.get(ev.branch_id)
        if prev and ev.step_id >= prev[0] and ev.timestamp_start + 1e-9 < prev[1]:
            errors.append(f"{ev.event_id}: non-monotonic timestamp in branch {ev.branch_id}")
        last_by_branch[ev.branch_id] = (ev.step_id, ev.timestamp_end)
    return errors


def load_trace_dir(path: str | Path) -> list[Trace]:
    p = Path(path)
    traces = [Trace.from_jsonl(x) for x in sorted(p.glob("*.jsonl"))]
    return traces


def merge_traces(traces: list[Trace], metadata: Optional[dict[str, Any]] = None) -> Trace:
    events: list[Event] = []
    for tr in traces:
        events.extend(tr.events)
    return Trace(metadata=metadata or {"merged_at": time.time(), "num_traces": len(traces)}, events=events)


def to_jsonl(trace: Trace, path: str | Path) -> None:
    trace.to_jsonl(path)


def from_jsonl(path: str | Path) -> Trace:
    return Trace.from_jsonl(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    args = ap.parse_args()
    trace = Trace.from_jsonl(args.trace)
    errors = validate_trace(trace)
    if errors:
        for err in errors:
            print(err)
        raise SystemExit(1)
    print(f"OK: {len(trace.events)} events")


if __name__ == "__main__":
    main()
