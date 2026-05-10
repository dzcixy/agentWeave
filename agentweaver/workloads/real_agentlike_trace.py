from __future__ import annotations

import argparse
import asyncio
import csv
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt

from agentweaver.profiling.async_vllm_client import AsyncVLLMClient
from agentweaver.tracing.prompt_segmenter import segment_prompt
from agentweaver.tracing.trace_schema import ContextSegmentRef, Event, Trace
from agentweaver.utils.hashing import prompt_hash, stable_hash
from agentweaver.utils.io import ensure_dir, write_csv
from agentweaver.utils.tokenization import count_tokens


def _messages(instance_id: str, branch_id: str, observation: str | None = None) -> list[dict[str, str]]:
    shared_repo = "\n".join(
        [
            "Repository file src/agent/router.py defines route_patch(branch, observation).",
            "Repository file tests/test_router.py asserts that failed tool logs trigger revision.",
            "Issue: a fork-join coding agent must produce a minimal patch after pytest feedback.",
        ]
    )
    messages = [
        {"role": "system", "content": "You are a local software engineering agent. Produce concise patch plans."},
        {"role": "user", "content": "Available tools: shell, python, pytest, grep, sed."},
        {"role": "user", "content": f"SWE-like controlled issue {instance_id}: fix failing branch behavior."},
        {"role": "user", "content": f"Repository context:\n```text\n{shared_repo}\n```"},
        {"role": "assistant", "content": f"Branch suffix {branch_id}: inspect likely failure and propose patch."},
    ]
    if observation is not None:
        messages.append({"role": "user", "content": f"Tool observation/test_log:\n{observation}"})
        messages.append({"role": "assistant", "content": f"Revised branch suffix {branch_id}: produce final patch after observation."})
    return messages


def _refs(seg_defs: list) -> list[ContextSegmentRef]:
    return [ContextSegmentRef(s.segment_id, s.segment_type, s.start_pos, s.length) for s in seg_defs]


def _tool_command(instance_idx: int, branch_idx: int) -> str:
    if branch_idx % 3 == 0:
        return "python -c \"import time; time.sleep(1); print('pytest failed: assertion branch needs revision')\""
    if branch_idx % 3 == 1:
        return "python -c \"import time; time.sleep(5); print('build completed with warning')\""
    return "python -c \"print('grep result: route_patch handles observation tokens')\""


async def _run(args: argparse.Namespace) -> list[Path]:
    out_dir = ensure_dir(args.out)
    client = AsyncVLLMClient(args.server, args.model, args.tokenizer_path)
    if not await client.health():
        raise RuntimeError(f"vLLM server unavailable: {args.server}")
    trace_paths: list[Path] = []
    summary_rows: list[dict[str, object]] = []
    for inst in range(args.instances):
        instance_id = f"real_agentlike_{inst:04d}"
        session_id = f"real-{stable_hash([instance_id, time.time()], 8)}"
        t0 = time.time()
        events: list[Event] = [
            Event(
                event_id=f"{instance_id}:fork",
                session_id=session_id,
                instance_id=instance_id,
                branch_id="root",
                parent_branch_id=None,
                step_id=0,
                node_id=f"{instance_id}:fork",
                node_type="fork",
                timestamp_ready=t0,
                timestamp_start=t0,
                timestamp_end=t0,
            )
        ]
        success_branch = args.branch_fanout // 2
        for b in range(args.branch_fanout):
            branch_id = f"b{b}"
            msgs0 = _messages(instance_id, branch_id)
            prompt0 = "\n".join(m["content"] for m in msgs0)
            seg0 = segment_prompt(msgs0, metadata={"model": args.model, "instance_id": instance_id, "branch_id": branch_id})
            rec0 = await client.chat(f"{instance_id}-{branch_id}-llm0", prompt0, args.max_tokens, stream=args.stream)
            if not rec0.status.startswith("ok"):
                raise RuntimeError(f"LLM0 failed for {instance_id}/{branch_id}: {rec0.error}")
            events.append(
                Event(
                    event_id=f"{instance_id}:{branch_id}:llm0",
                    session_id=session_id,
                    instance_id=instance_id,
                    branch_id=branch_id,
                    parent_branch_id="root",
                    step_id=1,
                    node_id=f"{instance_id}:{branch_id}:llm0",
                    node_type="llm",
                    role="worker",
                    model=args.model,
                    timestamp_ready=rec0.start_time,
                    timestamp_start=rec0.start_time,
                    timestamp_end=rec0.end_time,
                    latency=rec0.e2e_latency,
                    input_tokens=rec0.prompt_tokens_server or rec0.prompt_tokens_client,
                    output_tokens=rec0.completion_tokens_server or rec0.completion_tokens_client,
                    context_length=rec0.prompt_tokens_server or rec0.prompt_tokens_client,
                    prompt_hash=prompt_hash(prompt0),
                    shared_prefix_id=seg0[0].segment_id if seg0 else None,
                    context_segments=_refs(seg0),
                    context_segment_defs=seg0,
                    ttft=rec0.ttft,
                    tpot=rec0.tpot,
                )
            )
            command = _tool_command(inst, b)
            ts = time.time()
            cp = subprocess.run(command, shell=True, text=True, capture_output=True)
            te = time.time()
            observation = (cp.stdout or "") + (cp.stderr or "")
            events.append(
                Event(
                    event_id=f"{instance_id}:{branch_id}:tool0",
                    session_id=session_id,
                    instance_id=instance_id,
                    branch_id=branch_id,
                    parent_branch_id="root",
                    step_id=2,
                    node_id=f"{instance_id}:{branch_id}:tool0",
                    node_type="tool",
                    timestamp_ready=ts,
                    timestamp_start=ts,
                    timestamp_end=te,
                    latency=te - ts,
                    tool_type="pytest" if "pytest" in observation else "shell_other",
                    command=command,
                    tool_latency=te - ts,
                    observation_tokens=count_tokens(observation, args.model),
                    exit_code=cp.returncode,
                )
            )
            msgs1 = _messages(instance_id, branch_id, observation)
            prompt1 = "\n".join(m["content"] for m in msgs1)
            seg1 = segment_prompt(msgs1, metadata={"model": args.model, "instance_id": instance_id, "branch_id": branch_id})
            rec1 = await client.chat(f"{instance_id}-{branch_id}-llm1", prompt1, args.max_tokens, stream=args.stream)
            if not rec1.status.startswith("ok"):
                raise RuntimeError(f"LLM1 failed for {instance_id}/{branch_id}: {rec1.error}")
            events.append(
                Event(
                    event_id=f"{instance_id}:{branch_id}:llm1",
                    session_id=session_id,
                    instance_id=instance_id,
                    branch_id=branch_id,
                    parent_branch_id="root",
                    step_id=3,
                    node_id=f"{instance_id}:{branch_id}:llm1",
                    node_type="llm",
                    role="worker",
                    model=args.model,
                    timestamp_ready=rec1.start_time,
                    timestamp_start=rec1.start_time,
                    timestamp_end=rec1.end_time,
                    latency=rec1.e2e_latency,
                    input_tokens=rec1.prompt_tokens_server or rec1.prompt_tokens_client,
                    output_tokens=rec1.completion_tokens_server or rec1.completion_tokens_client,
                    context_length=rec1.prompt_tokens_server or rec1.prompt_tokens_client,
                    prompt_hash=prompt_hash(prompt1),
                    shared_prefix_id=seg1[0].segment_id if seg1 else None,
                    context_segments=_refs(seg1),
                    context_segment_defs=seg1,
                    ttft=rec1.ttft,
                    tpot=rec1.tpot,
                )
            )
            tv = time.time()
            success = b == success_branch
            events.append(
                Event(
                    event_id=f"{instance_id}:{branch_id}:verify0",
                    session_id=session_id,
                    instance_id=instance_id,
                    branch_id=branch_id,
                    parent_branch_id="root",
                    step_id=4,
                    node_id=f"{instance_id}:{branch_id}:verify0",
                    node_type="verifier",
                    timestamp_ready=tv,
                    timestamp_start=tv,
                    timestamp_end=tv + 0.001,
                    latency=0.001,
                    verifier_result="pass" if success else "fail",
                    success=success,
                    is_first_success=success,
                    patch_hash=stable_hash([instance_id, branch_id, "pseudo", success]),
                )
            )
        tj = time.time()
        events.append(
            Event(
                event_id=f"{instance_id}:join",
                session_id=session_id,
                instance_id=instance_id,
                branch_id="root",
                parent_branch_id=None,
                step_id=5,
                node_id=f"{instance_id}:join",
                node_type="join",
                timestamp_ready=tj,
                timestamp_start=tj,
                timestamp_end=tj,
                latency=0.0,
            )
        )
        trace = Trace(
            {
                "framework": "real_agentlike_trace",
                "benchmark": "real_agentlike_controlled",
                "model": args.model,
                "server": args.server,
                "timestamp": time.time(),
                "run_id": args.run_id,
                "pseudo_verifier": True,
            },
            sorted(events, key=lambda e: (e.timestamp_start, e.branch_id, e.step_id)),
        )
        path = out_dir / f"{instance_id}.jsonl"
        trace.to_jsonl(path)
        trace_paths.append(path)
        summary_rows.append(
            {
                "instance_id": instance_id,
                "run_id": args.run_id,
                "branch_fanout": args.branch_fanout,
                "llm_events": sum(1 for e in events if e.node_type == "llm"),
                "tool_events": sum(1 for e in events if e.node_type == "tool"),
                "total_llm_latency": sum(e.latency for e in events if e.node_type == "llm"),
                "total_tool_latency": sum(e.latency for e in events if e.node_type == "tool"),
                "pseudo_success_branch": f"b{success_branch}",
            }
        )
    ensure_dir("data/results")
    write_csv(args.summary_out, summary_rows)
    _plots(summary_rows, args.latency_plot_out, args.context_plot_out)
    return trace_paths


def _plots(rows: list[dict[str, object]], latency_plot_out: str, context_plot_out: str) -> None:
    ensure_dir(Path(latency_plot_out).parent)
    ensure_dir(Path(context_plot_out).parent)
    labels = [str(r["instance_id"]) for r in rows]
    llm = [float(r["total_llm_latency"]) for r in rows]
    tool = [float(r["total_tool_latency"]) for r in rows]
    plt.figure(figsize=(6, 3))
    plt.bar(labels, llm, label="LLM")
    plt.bar(labels, tool, bottom=llm, label="Tool")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("seconds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(latency_plot_out)
    plt.close()
    plt.figure(figsize=(5, 3))
    plt.plot(labels, [int(r["llm_events"]) for r in rows], marker="o", label="LLM events")
    plt.plot(labels, [int(r["tool_events"]) for r in rows], marker="o", label="Tool events")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(context_plot_out)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen-coder-7b")
    ap.add_argument("--tokenizer-path", default=None)
    ap.add_argument("--run-id", default="real_agentlike_pr2_v2")
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--branch-fanout", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--out", default="data/traces/real_agentlike_h100")
    ap.add_argument("--summary-out", default="data/results/real_agentlike_trace_summary.csv")
    ap.add_argument("--latency-plot-out", default="data/plots/real_agentlike_latency_breakdown.pdf")
    ap.add_argument("--context-plot-out", default="data/plots/real_agentlike_context_reuse.pdf")
    args = ap.parse_args()
    paths = asyncio.run(_run(args))
    print(f"wrote {len(paths)} real agent-like traces to {args.out}")


if __name__ == "__main__":
    main()
