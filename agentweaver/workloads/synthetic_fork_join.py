from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentweaver.tracing.prompt_segmenter import segment_prompt
from agentweaver.tracing.trace_schema import ContextSegmentRef, Event, Trace
from agentweaver.utils.hashing import prompt_hash, stable_hash
from agentweaver.utils.io import ensure_dir, read_yaml


@dataclass(frozen=True)
class SyntheticScenario:
    name: str
    shared_prefix_len: int
    branch_fanout: int
    tool_latency: float
    exact_prefix_reuse_ratio: float
    success_branch: str


FIXED_SCENARIOS: dict[str, SyntheticScenario] = {
    "S1_context_heavy": SyntheticScenario("S1_context_heavy", 8192, 4, 10, 0.75, "middle"),
    "S2_branch_heavy": SyntheticScenario("S2_branch_heavy", 4096, 8, 10, 0.5, "first"),
    "S3_tool_stall_heavy": SyntheticScenario("S3_tool_stall_heavy", 4096, 4, 60, 0.5, "middle"),
    "S4_low_reuse_negative": SyntheticScenario("S4_low_reuse_negative", 1024, 4, 10, 0.0, "middle"),
    "S5_tool_dominated_negative": SyntheticScenario("S5_tool_dominated_negative", 1024, 2, 300, 0.25, "last"),
}


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else [v]


def _pick(rng: random.Random, v: Any) -> Any:
    return rng.choice(_as_list(v))


def _success_index(branch_fanout: int, success_branch: str, rng: random.Random) -> int:
    if success_branch == "first":
        return 0
    if success_branch == "middle":
        return branch_fanout // 2
    if success_branch == "last":
        return branch_fanout - 1
    return rng.randrange(branch_fanout)


def _blob(prefix: str, tokens: int, modulo: int = 997) -> str:
    return " ".join(f"{prefix}_{i % modulo}" for i in range(max(0, tokens)))


def _branch_messages(
    instance_id: str,
    branch_id: str,
    shared_prefix_len: int,
    exact_prefix_reuse_ratio: float,
    observation_tokens: int = 0,
    revised: bool = False,
) -> list[dict[str, str]]:
    shared_tokens = int(shared_prefix_len * exact_prefix_reuse_ratio)
    private_tokens = max(32, shared_prefix_len - shared_tokens)
    repo_tokens = max(0, int(shared_tokens * 0.55))
    history_tokens = max(0, shared_tokens - repo_tokens)
    if exact_prefix_reuse_ratio <= 0:
        repo_blob = ""
        shared_history = ""
    else:
        repo_blob = _blob(f"{instance_id}_shared_repo", repo_tokens)
        shared_history = _blob(f"{instance_id}_shared_history", history_tokens)
    private_plan = _blob(f"{instance_id}_{branch_id}_private_plan", private_tokens, 521)
    messages = [
        {"role": "system", "content": "You are a coding agent. Produce minimal patches and run tests."},
        {
            "role": "user",
            "content": "Available tools: shell, grep, sed, python, pytest, build. Tool schema is stable for this run.",
        },
        {"role": "user", "content": f"SWE-bench issue: {instance_id}. Fix the failing behavior."},
        {"role": "user", "content": "Repository context:\n```python\n" + repo_blob + "\n```"},
        {"role": "user", "content": "Shared history:\n" + shared_history},
        {"role": "assistant", "content": f"Branch suffix {branch_id}: {private_plan}"},
    ]
    if observation_tokens:
        obs = _blob(f"{instance_id}_{branch_id}_pytest_failure", observation_tokens, 251)
        messages.append({"role": "user", "content": f"pytest failed with test_log stdout stderr:\n{obs}"})
    if revised:
        revised_plan = _blob(f"{instance_id}_{branch_id}_revised_patch", max(32, private_tokens // 3), 389)
        messages.append({"role": "assistant", "content": f"Revised branch suffix {branch_id}: {revised_plan}"})
    return messages


def _refs(seg_defs: list[Any]) -> list[ContextSegmentRef]:
    return [ContextSegmentRef(s.segment_id, s.segment_type, s.start_pos, s.length) for s in seg_defs]


def make_synthetic_trace(
    instance_id: str,
    branch_fanout: int = 4,
    shared_prefix_len: int = 4096,
    tool_latency: float = 10.0,
    output_tokens: int = 512,
    success_branch: str = "random",
    context_reuse_ratio: float = 0.75,
    exact_prefix_reuse_ratio: float | None = None,
    seed: int = 1,
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    scenario: str | None = None,
) -> Trace:
    exact_prefix_reuse_ratio = context_reuse_ratio if exact_prefix_reuse_ratio is None else exact_prefix_reuse_ratio
    rng = random.Random(f"{seed}:{instance_id}:{scenario or 'synthetic'}")
    session_id = f"syn-{stable_hash([instance_id, scenario], 8)}"
    start = 1_700_000_000.0 + rng.random()
    success_idx = _success_index(branch_fanout, success_branch, rng)
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
    verifier_end_times: list[float] = []
    for b in range(branch_fanout):
        branch_id = f"b{b}"
        observation_tokens = 192 + rng.randrange(0, 256)
        msgs0 = _branch_messages(instance_id, branch_id, shared_prefix_len, exact_prefix_reuse_ratio)
        seg0 = segment_prompt(msgs0, metadata={"model": model, "instance_id": instance_id, "branch_id": branch_id})
        input0 = sum(s.length for s in seg0)
        llm0_start = start + 0.1 + b * 0.02
        out0 = max(32, output_tokens + rng.randrange(-32, 33))
        llm0_lat = 0.02 + input0 * 0.00009 + out0 * 0.0015 + rng.random() * 0.01
        prompt0 = "\n".join(m["content"] for m in msgs0)
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
                model=model,
                timestamp_ready=start,
                timestamp_start=llm0_start,
                timestamp_end=llm0_start + llm0_lat,
                latency=llm0_lat,
                input_tokens=input0,
                output_tokens=out0,
                context_length=input0,
                prompt_hash=prompt_hash(prompt0),
                shared_prefix_id=seg0[0].segment_id if seg0 else None,
                context_segments=_refs(seg0),
                context_segment_defs=seg0,
                prefill_latency=input0 * 0.00009,
                decode_latency=out0 * 0.0015,
            )
        )
        tool_start = llm0_start + llm0_lat + 0.005
        branch_tool_latency = max(0.0, tool_latency * rng.uniform(0.75, 1.35))
        tool_type = "pytest" if b % 3 != 1 else ("build" if b % 3 == 1 else "shell_other")
        passed = b == success_idx
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
                timestamp_ready=tool_start,
                timestamp_start=tool_start,
                timestamp_end=tool_start + branch_tool_latency,
                latency=branch_tool_latency,
                tool_type=tool_type,
                command="pytest -q" if tool_type == "pytest" else "python -m build",
                tool_latency=branch_tool_latency,
                observation_tokens=observation_tokens,
                exit_code=0 if passed else 1,
            )
        )
        msgs1 = _branch_messages(
            instance_id,
            branch_id,
            shared_prefix_len,
            exact_prefix_reuse_ratio,
            observation_tokens=observation_tokens,
            revised=True,
        )
        seg1 = segment_prompt(msgs1, metadata={"model": model, "instance_id": instance_id, "branch_id": branch_id})
        input1 = sum(s.length for s in seg1)
        llm1_start = tool_start + branch_tool_latency + 0.005
        out1 = max(32, output_tokens // 2 + rng.randrange(-16, 17))
        llm1_lat = 0.02 + input1 * 0.00009 + out1 * 0.0015 + rng.random() * 0.01
        prompt1 = "\n".join(m["content"] for m in msgs1)
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
                model=model,
                timestamp_ready=llm1_start,
                timestamp_start=llm1_start,
                timestamp_end=llm1_start + llm1_lat,
                latency=llm1_lat,
                input_tokens=input1,
                output_tokens=out1,
                context_length=input1,
                prompt_hash=prompt_hash(prompt1),
                shared_prefix_id=seg1[0].segment_id if seg1 else None,
                context_segments=_refs(seg1),
                context_segment_defs=seg1,
                prefill_latency=input1 * 0.00009,
                decode_latency=out1 * 0.0015,
            )
        )
        verify_start = llm1_start + llm1_lat + 0.005
        verifier_end_times.append(verify_start + 0.01)
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
                timestamp_ready=verify_start,
                timestamp_start=verify_start,
                timestamp_end=verify_start + 0.01,
                latency=0.01,
                verifier_result="pass" if passed else "fail",
                patch_id=f"{instance_id}-{branch_id}",
                patch_hash=stable_hash([instance_id, branch_id, passed, "final"]),
                is_first_success=passed,
                success=passed,
            )
        )
    join_t = max(verifier_end_times or [start])
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
            timestamp_ready=join_t,
            timestamp_start=join_t,
            timestamp_end=join_t,
            latency=0.0,
            success=True,
        )
    )
    metadata = {
        "hardware": "synthetic",
        "model": model,
        "framework": "synthetic_fork_join",
        "benchmark": "synthetic",
        "timestamp": time.time(),
        "scenario": scenario or instance_id,
        "branch_fanout": branch_fanout,
        "shared_prefix_len": shared_prefix_len,
        "exact_prefix_reuse_ratio": exact_prefix_reuse_ratio,
        "success_branch": success_branch,
    }
    return Trace(metadata, sorted(events, key=lambda e: (e.timestamp_start, e.branch_id, e.step_id)))


def make_scenario_trace(scenario: SyntheticScenario, seed: int = 1) -> Trace:
    return make_synthetic_trace(
        instance_id=scenario.name,
        branch_fanout=scenario.branch_fanout,
        shared_prefix_len=scenario.shared_prefix_len,
        tool_latency=scenario.tool_latency,
        success_branch=scenario.success_branch,
        exact_prefix_reuse_ratio=scenario.exact_prefix_reuse_ratio,
        context_reuse_ratio=scenario.exact_prefix_reuse_ratio,
        seed=seed,
        scenario=scenario.name,
    )


def generate_fixed_scenarios(out_dir: str | Path = "data/traces/synthetic", seed: int = 1) -> list[Path]:
    out = ensure_dir(out_dir)
    paths: list[Path] = []
    for scenario in FIXED_SCENARIOS.values():
        trace = make_scenario_trace(scenario, seed=seed)
        path = out / f"{scenario.name}.jsonl"
        trace.to_jsonl(path)
        paths.append(path)
    return paths


def generate_from_config(config_path: str | Path, out_dir: str | Path, run_id: str = "small") -> list[Path]:
    cfg = read_yaml(config_path)
    if cfg.get("fixed_scenarios", False):
        return generate_fixed_scenarios(Path(out_dir) / run_id, int(cfg.get("seed", 1)))
    out = ensure_dir(Path(out_dir) / run_id)
    rng = random.Random(int(cfg.get("seed", 1)))
    n = int(cfg.get("num_instances", 5))
    paths: list[Path] = []
    for i in range(n):
        exact = float(_pick(rng, cfg.get("exact_prefix_reuse_ratio", cfg.get("context_reuse_ratio", 0.75))))
        trace = make_synthetic_trace(
            instance_id=f"synthetic_{i:04d}",
            branch_fanout=int(_pick(rng, cfg.get("branch_fanout", 4))),
            shared_prefix_len=int(_pick(rng, cfg.get("shared_prefix_len", 4096))),
            tool_latency=float(_pick(rng, cfg.get("tool_latency", 10))),
            output_tokens=int(_pick(rng, cfg.get("output_tokens", 512))),
            success_branch=str(cfg.get("success_branch", "random")),
            context_reuse_ratio=exact,
            exact_prefix_reuse_ratio=exact,
            seed=int(cfg.get("seed", 1)),
        )
        path = out / f"{trace.events[0].instance_id}.jsonl"
        trace.to_jsonl(path)
        paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="data/traces/synthetic")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--fixed-scenarios", action="store_true")
    args = ap.parse_args()
    if args.fixed_scenarios or args.config is None:
        paths = generate_fixed_scenarios(args.out_dir)
        print(f"wrote {len(paths)} fixed scenarios under {args.out_dir}")
        return
    paths = generate_from_config(args.config, args.out_dir, args.run_id or "small")
    print(f"wrote {len(paths)} traces under {Path(args.out_dir) / (args.run_id or 'small')}")


if __name__ == "__main__":
    main()
