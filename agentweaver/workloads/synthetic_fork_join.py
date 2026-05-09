from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

from agentweaver.tracing.prompt_segmenter import segment_prompt
from agentweaver.tracing.trace_schema import ContextSegmentRef, Event, Trace
from agentweaver.utils.hashing import prompt_hash, stable_hash
from agentweaver.utils.io import ensure_dir, read_yaml


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else [v]


def _pick(rng: random.Random, v: Any) -> Any:
    return rng.choice(_as_list(v))


def _messages(instance_id: str, branch_id: str, shared_prefix_len: int, reuse_ratio: float) -> list[dict[str, str]]:
    repo_tokens = max(32, int(shared_prefix_len * reuse_ratio))
    history_tokens = max(16, shared_prefix_len - repo_tokens)
    repo_blob = " ".join([f"repo_symbol_{i % 97}" for i in range(repo_tokens)])
    hist_blob = " ".join([f"history_turn_{i % 29}" for i in range(history_tokens)])
    return [
        {"role": "system", "content": "You are a coding agent. Use available tools and produce patches."},
        {"role": "user", "content": f"SWE-bench issue: {instance_id}. Fix the failing test without unrelated changes."},
        {"role": "user", "content": "Repository context:\n```python\n" + repo_blob + "\n```"},
        {"role": "assistant", "content": f"Prior shared history: {hist_blob}"},
        {"role": "assistant", "content": f"Branch plan for {branch_id}: inspect, patch, run pytest."},
    ]


def make_synthetic_trace(
    instance_id: str,
    branch_fanout: int = 4,
    shared_prefix_len: int = 4096,
    tool_latency: float = 10.0,
    output_tokens: int = 512,
    success_branch: str = "random",
    context_reuse_ratio: float = 0.75,
    seed: int = 1,
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
) -> Trace:
    rng = random.Random(f"{seed}:{instance_id}")
    session_id = f"syn-{stable_hash(instance_id, 8)}"
    start = 1_700_000_000.0 + rng.random()
    if success_branch == "first":
        success_idx = 0
    elif success_branch == "middle":
        success_idx = branch_fanout // 2
    elif success_branch == "last":
        success_idx = branch_fanout - 1
    else:
        success_idx = rng.randrange(branch_fanout)
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
            timestamp_ready=start,
            timestamp_start=start,
            timestamp_end=start,
        )
    ]
    for b in range(branch_fanout):
        branch_id = f"b{b}"
        msgs = _messages(instance_id, branch_id, shared_prefix_len, context_reuse_ratio)
        seg_defs = segment_prompt(msgs, metadata={"model": model, "instance_id": instance_id, "branch_id": branch_id})
        refs = [ContextSegmentRef(s.segment_id, s.segment_type, s.start_pos, s.length) for s in seg_defs]
        in_tokens = sum(s.length for s in seg_defs)
        llm_start = start + 0.1 + b * 0.03
        llm_lat = 0.02 + in_tokens * 0.00009 + output_tokens * 0.0015 + rng.random() * 0.02
        prompt_txt = "\n".join(m["content"] for m in msgs)
        events.append(
            Event(
                event_id=f"{instance_id}:{branch_id}:llm0",
                session_id=session_id,
                instance_id=instance_id,
                branch_id=branch_id,
                parent_branch_id="root",
                step_id=1,
                node_id=f"{branch_id}:llm0",
                node_type="llm",
                role="worker",
                model=model,
                timestamp_ready=start,
                timestamp_start=llm_start,
                timestamp_end=llm_start + llm_lat,
                latency=llm_lat,
                input_tokens=in_tokens,
                output_tokens=output_tokens + rng.randrange(-32, 33),
                context_length=in_tokens,
                prompt_hash=prompt_hash(prompt_txt),
                shared_prefix_id=seg_defs[0].segment_id,
                context_segments=refs,
                context_segment_defs=seg_defs,
                prefill_latency=in_tokens * 0.00009,
                decode_latency=output_tokens * 0.0015,
            )
        )
        t_start = llm_start + llm_lat + 0.01
        branch_tool_latency = max(0.0, tool_latency * rng.uniform(0.5, 1.8))
        events.append(
            Event(
                event_id=f"{instance_id}:{branch_id}:tool0",
                session_id=session_id,
                instance_id=instance_id,
                branch_id=branch_id,
                parent_branch_id="root",
                step_id=2,
                node_id=f"{branch_id}:tool0",
                node_type="tool",
                timestamp_ready=t_start,
                timestamp_start=t_start,
                timestamp_end=t_start + branch_tool_latency,
                latency=branch_tool_latency,
                tool_type="pytest",
                command="pytest -q",
                tool_latency=branch_tool_latency,
                observation_tokens=128 + rng.randrange(0, 256),
                exit_code=0 if b == success_idx else 1,
            )
        )
        v_start = t_start + branch_tool_latency + 0.005
        passed = b == success_idx
        events.append(
            Event(
                event_id=f"{instance_id}:{branch_id}:verify0",
                session_id=session_id,
                instance_id=instance_id,
                branch_id=branch_id,
                parent_branch_id="root",
                step_id=3,
                node_id=f"{branch_id}:verify0",
                node_type="verifier",
                timestamp_ready=v_start,
                timestamp_start=v_start,
                timestamp_end=v_start + 0.01,
                latency=0.01,
                verifier_result="pass" if passed else "fail",
                patch_id=f"{instance_id}-{branch_id}",
                patch_hash=stable_hash([instance_id, branch_id, passed]),
                is_first_success=passed,
                success=passed,
            )
        )
    metadata = {
        "hardware": "synthetic",
        "model": model,
        "framework": "synthetic_fork_join",
        "benchmark": "synthetic",
        "timestamp": time.time(),
        "branch_fanout": branch_fanout,
        "shared_prefix_len": shared_prefix_len,
    }
    return Trace(metadata, sorted(events, key=lambda e: (e.timestamp_start, e.branch_id, e.step_id)))


def generate_from_config(config_path: str | Path, out_dir: str | Path, run_id: str = "small") -> list[Path]:
    cfg = read_yaml(config_path)
    out = ensure_dir(Path(out_dir) / run_id)
    rng = random.Random(int(cfg.get("seed", 1)))
    n = int(cfg.get("num_instances", 5))
    paths: list[Path] = []
    for i in range(n):
        trace = make_synthetic_trace(
            instance_id=f"synthetic_{i:04d}",
            branch_fanout=int(_pick(rng, cfg.get("branch_fanout", 4))),
            shared_prefix_len=int(_pick(rng, cfg.get("shared_prefix_len", 4096))),
            tool_latency=float(_pick(rng, cfg.get("tool_latency", 10))),
            output_tokens=int(_pick(rng, cfg.get("output_tokens", 512))),
            success_branch=str(cfg.get("success_branch", "random")),
            context_reuse_ratio=float(_pick(rng, cfg.get("context_reuse_ratio", 0.75))),
            seed=int(cfg.get("seed", 1)),
        )
        path = out / f"{trace.events[0].instance_id}.jsonl"
        trace.to_jsonl(path)
        paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small_sanity.yaml")
    ap.add_argument("--out-dir", default="data/traces")
    ap.add_argument("--run-id", default="small")
    args = ap.parse_args()
    paths = generate_from_config(args.config, args.out_dir, args.run_id)
    print(f"wrote {len(paths)} traces under {Path(args.out_dir) / args.run_id}")


if __name__ == "__main__":
    main()
