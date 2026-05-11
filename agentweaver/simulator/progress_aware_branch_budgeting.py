from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.tracing.trace_schema import Event, Trace, load_trace_dir
from agentweaver.utils.io import ensure_dir, read_yaml, write_csv


BUDGET_POLICIES = ["launch_all", "fcfs_budget", "pabb_budget"]


@dataclass
class BranchSignals:
    instance_id: str
    branch_id: str
    source: str
    patch_nonempty: bool
    git_diff_bytes: int
    patch_hash: str
    pytest_command_seen: bool
    tool_returncode_success_or_progress: bool
    test_log_error_count: int
    duplicate_patch_hash: bool
    no_file_modification: bool
    steps_used: int
    llm_steps_used: int
    llm_tokens_used: int
    observation_tokens: int
    total_tool_time: float
    model_side_time: float
    measured_time_to_patch: float | None
    tokens_to_patch: int | None
    cost_to_patch: float | None
    official_verifier_result: str


def _event_time(ev: Event) -> float:
    return ev.timestamp_start or ev.timestamp_ready or 0.0


def _event_end(ev: Event) -> float:
    return ev.timestamp_end or _event_time(ev)


def _contains_test_command(command: str) -> bool:
    s = command.lower()
    return any(x in s for x in ("pytest", "tox", " runtests", " test ", " manage.py test", "unittest"))


def _error_count(command: str) -> int:
    s = command.lower()
    return sum(s.count(x.lower()) for x in ("FAIL", "ERROR", "Traceback", "AssertionError"))


def _patch_hash_from_events(events: list[Event]) -> tuple[str, int]:
    for ev in events:
        if ev.node_type == "verifier" and ev.patch_hash:
            tokens = sum(seg.length for seg in ev.context_segments if seg.segment_type == "patch")
            return ev.patch_hash, tokens
    for ev in events:
        for seg in ev.context_segments:
            if seg.segment_type == "patch" and seg.length > 0:
                payload = f"{ev.instance_id}:{ev.branch_id}:{seg.segment_id}:{seg.length}"
                return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], seg.length
    return "", 0


def _first_patch_point(events: list[Event]) -> tuple[float | None, int | None, float | None]:
    verifiers = [e for e in events if e.node_type == "verifier" and (e.patch_hash or any(seg.segment_type == "patch" and seg.length > 0 for seg in e.context_segments))]
    if not verifiers:
        return None, None, None
    patch_ev = min(verifiers, key=_event_time)
    start = min((_event_time(e) for e in events if _event_time(e)), default=0.0)
    cutoff = _event_time(patch_ev)
    tokens = sum(e.input_tokens + e.output_tokens for e in events if e.node_type == "llm" and _event_time(e) <= cutoff)
    tool = sum(float(e.tool_latency if e.tool_latency is not None else e.latency or 0.0) for e in events if e.node_type == "tool" and _event_time(e) <= cutoff)
    return max(0.0, cutoff - start) if start else None, tokens, tool


def extract_branch_signals(
    trace_dirs: list[str | Path],
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
) -> list[BranchSignals]:
    lm = LatencyModel.load(model_json)
    branches: dict[tuple[str, str, str], list[Event]] = defaultdict(list)
    for trace_dir in trace_dirs:
        for trace in load_trace_dir(trace_dir):
            source = str(trace.metadata.get("source", trace.metadata.get("source_trace_dir", trace_dir)))
            for ev in trace.events:
                if ev.branch_id == "root":
                    continue
                branches[(ev.instance_id, ev.branch_id, source)].append(ev)
    patch_counts: Counter[str] = Counter()
    patch_by_branch: dict[tuple[str, str, str], tuple[str, int]] = {}
    for key, events in branches.items():
        patch_hash, patch_tokens = _patch_hash_from_events(events)
        patch_by_branch[key] = (patch_hash, patch_tokens)
        if patch_hash:
            patch_counts[patch_hash] += 1

    rows: list[BranchSignals] = []
    for key, events in sorted(branches.items()):
        instance_id, branch_id, source = key
        llm = [e for e in events if e.node_type == "llm"]
        tools = [e for e in events if e.node_type == "tool"]
        verifiers = [e for e in events if e.node_type == "verifier"]
        patch_hash, patch_tokens = patch_by_branch[key]
        patch_nonempty = bool(patch_hash or patch_tokens > 0)
        commands = "\n".join(e.command or "" for e in tools)
        pytest_seen = any(_contains_test_command(e.command or "") for e in tools)
        rc_progress = any((e.exit_code == 0) for e in tools if e.exit_code is not None)
        official = "unknown"
        for ev in verifiers:
            if ev.verifier_result in {"pass", "fail"}:
                official = ev.verifier_result
                break
        time_to_patch, tokens_to_patch, tool_to_patch = _first_patch_point(events)
        model_time = sum(lm.predict_prefill(e.input_tokens) + lm.predict_decode(e.context_length or e.input_tokens, e.output_tokens) for e in llm)
        total_tool = sum(float(e.tool_latency if e.tool_latency is not None else e.latency or 0.0) for e in tools)
        rows.append(
            BranchSignals(
                instance_id=instance_id,
                branch_id=branch_id,
                source=source,
                patch_nonempty=patch_nonempty,
                git_diff_bytes=patch_tokens,
                patch_hash=patch_hash,
                pytest_command_seen=pytest_seen,
                tool_returncode_success_or_progress=rc_progress,
                test_log_error_count=_error_count(commands),
                duplicate_patch_hash=bool(patch_hash and patch_counts[patch_hash] > 1),
                no_file_modification=not patch_nonempty,
                steps_used=len([e for e in events if e.node_type in {"llm", "tool"}]),
                llm_steps_used=len(llm),
                llm_tokens_used=sum(e.input_tokens + e.output_tokens for e in llm),
                observation_tokens=sum(e.observation_tokens or 0 for e in tools),
                total_tool_time=total_tool,
                model_side_time=model_time,
                measured_time_to_patch=time_to_patch,
                tokens_to_patch=tokens_to_patch,
                cost_to_patch=(model_time + (tool_to_patch or 0.0)) if time_to_patch is not None else None,
                official_verifier_result=official,
            )
        )
    return rows


def _weights(path: str | Path) -> dict[str, float]:
    data = read_yaml(path) if Path(path).exists() else {}
    raw = data.get("pabb", data)
    return {
        "patch_nonempty": float(raw.get("patch_nonempty", 4.0)),
        "pytest_command_seen": float(raw.get("pytest_command_seen", 1.5)),
        "tool_returncode_success_or_progress": float(raw.get("tool_returncode_success_or_progress", 0.8)),
        "duplicate_patch_hash": float(raw.get("duplicate_patch_hash", 2.0)),
        "no_file_modification": float(raw.get("no_file_modification", 1.5)),
        "repeated_failure": float(raw.get("repeated_failure", 1.0)),
        "normalized_token_cost": float(raw.get("normalized_token_cost", 1.0)),
        "excessive_steps": float(raw.get("excessive_steps", 0.5)),
    }


def _utility(sig: BranchSignals, weights: dict[str, float], max_tokens: int, max_steps: int) -> float:
    token_cost = sig.llm_tokens_used / max(1, max_tokens)
    excessive = max(0, sig.llm_steps_used - max_steps) / max(1, max_steps)
    repeated_failure = 1.0 if sig.test_log_error_count > 0 and not sig.tool_returncode_success_or_progress else 0.0
    return (
        weights["patch_nonempty"] * float(sig.patch_nonempty)
        + weights["pytest_command_seen"] * float(sig.pytest_command_seen)
        + weights["tool_returncode_success_or_progress"] * float(sig.tool_returncode_success_or_progress)
        - weights["duplicate_patch_hash"] * float(sig.duplicate_patch_hash)
        - weights["no_file_modification"] * float(sig.no_file_modification)
        - weights["repeated_failure"] * repeated_failure
        - weights["normalized_token_cost"] * token_cost
        - weights["excessive_steps"] * excessive
    )


def _select(signals: list[BranchSignals], policy: str, max_active: int, max_steps: int, weights: dict[str, float]) -> list[BranchSignals]:
    if policy == "launch_all":
        return signals
    if policy == "fcfs_budget":
        return signals[:max_active]
    max_tokens = max([s.llm_tokens_used for s in signals] or [1])
    ranked = sorted(signals, key=lambda s: _utility(s, weights, max_tokens, max_steps), reverse=True)
    return ranked[:max_active]


def run_pabb(
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    weights_yaml: str | Path = "configs/pabb_weights.yaml",
    out_csv: str | Path = "data/results/pabb_branch_budget_pr4_algo.csv",
) -> list[dict[str, Any]]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    signals = extract_branch_signals(trace_dirs, model_json)
    weights = _weights(weights_yaml)
    by_instance: dict[str, list[BranchSignals]] = defaultdict(list)
    for sig in signals:
        by_instance[sig.instance_id].append(sig)
    rows: list[dict[str, Any]] = []
    for instance_id, inst_signals in sorted(by_instance.items()):
        inst_signals = sorted(inst_signals, key=lambda s: (s.source, s.branch_id))
        for policy in BUDGET_POLICIES:
            for max_active in [1, 2, 4]:
                for max_steps in [5, 10, 15]:
                    selected = _select(inst_signals, policy, max_active, max_steps, weights)
                    reachable_patch = [
                        s
                        for s in selected
                        if s.patch_nonempty and s.measured_time_to_patch is not None and s.llm_steps_used <= max_steps
                    ]
                    first = min(reachable_patch, key=lambda s: s.measured_time_to_patch or math.inf) if reachable_patch else None
                    duplicate_count = sum(1 for s in selected if s.duplicate_patch_hash)
                    official = next((s.official_verifier_result for s in selected if s.official_verifier_result in {"pass", "fail"}), "unknown")
                    rows.append(
                        {
                            "instance_id": instance_id,
                            "budget_policy": policy,
                            "max_active_branches": max_active,
                            "max_steps_per_branch": max_steps,
                            "time_to_first_nonempty_patch": "" if first is None else first.measured_time_to_patch,
                            "tokens_to_first_nonempty_patch": "" if first is None else first.tokens_to_patch,
                            "cost_to_first_nonempty_patch": "" if first is None else first.cost_to_patch,
                            "duplicate_patch_rate": duplicate_count / max(1, len(selected)),
                            "branches_pruned": max(0, len(inst_signals) - len(selected)),
                            "official_success_if_available": official,
                            "verifier_unknown_count": sum(1 for s in selected if s.official_verifier_result == "unknown"),
                            "total_tokens_used": sum(min(s.llm_tokens_used, int(s.llm_tokens_used * min(1.0, max_steps / max(1, s.llm_steps_used)))) for s in selected),
                            "total_tool_time": sum(s.total_tool_time * min(1.0, max_steps / max(1, s.llm_steps_used)) for s in selected),
                            "model_side_time": sum(s.model_side_time * min(1.0, max_steps / max(1, s.llm_steps_used)) for s in selected),
                            "selected_branches": len(selected),
                            "patch_nonempty_selected": sum(1 for s in selected if s.patch_nonempty),
                        }
                    )
    write_csv(out_csv, rows)
    plot_pabb(rows)
    return rows


def _metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _aggregate_for_plot(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    vals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = _metric(row, metric)
        if val is not None:
            vals[str(row["budget_policy"])].append(val)
    return {k: sum(v) / len(v) for k, v in vals.items() if v}


def _bar(values: dict[str, float], ylabel: str, out: str | Path) -> None:
    ensure_dir(Path(out).parent)
    labels = BUDGET_POLICIES
    ys = [values.get(x, 0.0) for x in labels]
    plt.figure(figsize=(5.8, 3.6))
    plt.bar(labels, ys)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_pabb(rows: list[dict[str, Any]]) -> None:
    _bar(_aggregate_for_plot(rows, "cost_to_first_nonempty_patch"), "cost to first non-empty patch", "data/plots/pabb_cost_to_patch_pr4_algo.pdf")
    token_vals = _aggregate_for_plot(rows, "tokens_to_first_nonempty_patch")
    dup_vals = _aggregate_for_plot(rows, "duplicate_patch_rate")
    ensure_dir("data/plots")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    labels = BUDGET_POLICIES
    axes[0].bar(labels, [token_vals.get(x, 0.0) for x in labels])
    axes[0].set_ylabel("tokens to first non-empty patch")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(labels, [dup_vals.get(x, 0.0) for x in labels])
    axes[1].set_ylabel("duplicate patch rate")
    axes[1].tick_params(axis="x", rotation=20)
    fig.savefig("data/plots/pabb_branch_budget_pr4_algo.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dirs", default="data/traces/mini_swe_lite10_r4_timed,data/traces/mini_swe_lite5_patchcap_verified")
    ap.add_argument("--model-json", default="data/profiles/h100_latency_model_pr2_v2.json")
    ap.add_argument("--weights-yaml", default="configs/pabb_weights.yaml")
    ap.add_argument("--out", default="data/results/pabb_branch_budget_pr4_algo.csv")
    args = ap.parse_args()
    rows = run_pabb([x for x in args.trace_dirs.split(",") if x.strip()], args.model_json, args.weights_yaml, args.out)
    print(json.dumps({"rows": len(rows), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
