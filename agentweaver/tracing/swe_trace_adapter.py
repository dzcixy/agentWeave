from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from agentweaver.tracing.prompt_segmenter import segment_prompt
from agentweaver.tracing.tool_wrapper import classify_command
from agentweaver.tracing.trace_schema import ContextSegmentRef, Event, Trace
from agentweaver.utils.hashing import prompt_hash, stable_hash
from agentweaver.utils.tokenization import count_tokens


def _load(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def convert_swe_traj(path: str | Path, out: str | Path | None = None, model: str = "Qwen/Qwen2.5-Coder-7B-Instruct") -> Trace:
    obj = _load(path)
    steps = obj.get("trajectory") or obj.get("steps") or obj.get("history") or []
    instance_id = obj.get("instance_id") or Path(path).stem.replace(".traj", "")
    session_id = obj.get("run_id") or stable_hash(str(path))
    events: list[Event] = []
    now = float(obj.get("start_time", time.time()))
    branch_id = obj.get("branch_id", "b0")
    for i, step in enumerate(steps):
        action = step.get("action") or step.get("thought") or step.get("prompt") or ""
        obs = step.get("observation") or step.get("output") or ""
        start = float(step.get("timestamp_start", step.get("start_time", now + i)))
        end = float(step.get("timestamp_end", step.get("end_time", start + float(step.get("latency", 0.1)))))
        if step.get("messages") or step.get("prompt"):
            msgs = step.get("messages") or [{"role": "user", "content": step.get("prompt", action)}]
            rendered = "\n".join(str(m.get("content", "")) for m in msgs)
            segs = segment_prompt(msgs, metadata={"model": model, "instance_id": instance_id, "branch_id": branch_id})
            events.append(
                Event(
                    event_id=f"{instance_id}:{branch_id}:llm{i}",
                    session_id=session_id,
                    instance_id=instance_id,
                    branch_id=branch_id,
                    parent_branch_id=None,
                    step_id=2 * i,
                    node_id=f"{branch_id}:llm{i}",
                    node_type="llm",
                    role="worker",
                    model=model,
                    timestamp_ready=start,
                    timestamp_start=start,
                    timestamp_end=end,
                    latency=end - start,
                    input_tokens=count_tokens(rendered, model),
                    output_tokens=count_tokens(str(step.get("response", "")), model),
                    context_length=count_tokens(rendered, model),
                    prompt_hash=prompt_hash(rendered),
                    shared_prefix_id=segs[0].segment_id if segs else None,
                    context_segments=[ContextSegmentRef(s.segment_id, s.segment_type, s.start_pos, s.length) for s in segs],
                    context_segment_defs=segs,
                )
            )
        cmd = step.get("command") or step.get("tool_call") or (action if isinstance(action, str) and action.startswith(("pytest", "grep", "sed", "cat")) else None)
        if cmd:
            t0 = end
            t1 = float(step.get("tool_end_time", t0 + float(step.get("tool_latency", 0.1))))
            events.append(
                Event(
                    event_id=f"{instance_id}:{branch_id}:tool{i}",
                    session_id=session_id,
                    instance_id=instance_id,
                    branch_id=branch_id,
                    parent_branch_id=None,
                    step_id=2 * i + 1,
                    node_id=f"{branch_id}:tool{i}",
                    node_type="tool",
                    timestamp_ready=t0,
                    timestamp_start=t0,
                    timestamp_end=t1,
                    latency=t1 - t0,
                    tool_type=classify_command(str(cmd)),
                    command=str(cmd),
                    tool_latency=t1 - t0,
                    observation_tokens=count_tokens(str(obs)),
                    exit_code=step.get("exit_code"),
                )
            )
    success = bool(obj.get("success", obj.get("resolved", False)))
    if events:
        t = max(e.timestamp_end for e in events)
        events.append(
            Event(
                event_id=f"{instance_id}:{branch_id}:verify",
                session_id=session_id,
                instance_id=instance_id,
                branch_id=branch_id,
                parent_branch_id=None,
                step_id=9999,
                node_id=f"{branch_id}:verify",
                node_type="verifier",
                timestamp_ready=t,
                timestamp_start=t,
                timestamp_end=t + 0.001,
                latency=0.001,
                verifier_result="pass" if success else "fail",
                success=success,
                patch_hash=stable_hash(obj.get("patch", "")),
            )
        )
    tr = Trace({"framework": "swe-agent-adapter", "source": str(path), "timestamp": time.time()}, events)
    if out:
        tr.to_jsonl(out)
    return tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tr = convert_swe_traj(args.traj, args.out)
    print(f"wrote {len(tr.events)} events to {args.out}")


if __name__ == "__main__":
    main()
