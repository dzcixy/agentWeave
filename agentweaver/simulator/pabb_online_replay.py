from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.tracing.trace_schema import Event, load_trace_dir
from agentweaver.utils.io import ensure_dir, read_yaml, write_csv


ONLINE_POLICIES = ["launch_all", "fcfs_budget", "pabb_online", "pabb_oracle_upper_bound"]
SNAPSHOT_POLICIES = ["fcfs_budget", "pabb_online_v3", "pabb_snapshot_online", "pabb_oracle_upper_bound"]


@dataclass
class BranchOnlineState:
    branch_id: str
    events: list[Event]
    current_step_index: int = 0
    llm_tokens_seen: int = 0
    tool_outputs_seen: int = 0
    pytest_seen: bool = False
    latest_returncode: int | None = None
    error_count_seen: int = 0
    file_modification_seen: bool = False
    patch_snapshot_available: bool = False
    modified_files_count: int = 0
    untracked_files_count: int = 0
    git_diff_stat_bytes: int = 0
    patch_candidate_seen: bool = False
    patch_hash_seen: str = ""
    duplicate_patch_seen_so_far: bool = False
    official_verifier_seen: str = "unknown"
    steps_executed: int = 0
    last_progress_time: int = 0
    alive: bool = True
    pruned: bool = False
    no_progress_steps: int = 0
    age: int = 0
    model_side_time: float = 0.0
    tool_time: float = 0.0
    observation_tokens: int = 0

    def done(self) -> bool:
        return self.current_step_index >= len(self.events) or not self.alive

    def next_event(self) -> Event | None:
        return None if self.done() else self.events[self.current_step_index]


def _event_time(ev: Event) -> float:
    return ev.timestamp_start or ev.timestamp_ready or float(ev.step_id)


def _contains_test_command(command: str | None) -> bool:
    s = (command or "").lower()
    return any(x in s for x in ("pytest", "tox", " runtests", " test ", " manage.py test", "unittest"))


def _looks_like_file_write(command: str | None) -> bool:
    s = (command or "").lower()
    return any(x in s for x in ("sed -i", "cat >", "python - <<", "perl -", "apply_patch", "tee ", "mv ", "cp "))


def _error_count(text: str | None) -> int:
    s = (text or "").lower()
    return sum(s.count(x.lower()) for x in ("FAIL", "ERROR", "Traceback", "AssertionError"))


def _event_patch(ev: Event) -> tuple[bool, str]:
    if ev.patch_hash:
        return True, ev.patch_hash
    for seg in ev.context_segments:
        if seg.segment_type == "patch" and seg.length > 0:
            payload = f"{ev.instance_id}:{ev.branch_id}:{seg.segment_id}:{seg.length}"
            return True, hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return False, ""


def _future_patch_value(events: list[Event]) -> tuple[int, int]:
    patch = 0
    first_tokens = 10**12
    tokens = 0
    for ev in events:
        if ev.node_type == "llm":
            tokens += ev.input_tokens + ev.output_tokens
        has_patch, _ = _event_patch(ev)
        if has_patch:
            patch = 1
            first_tokens = min(first_tokens, max(1, tokens))
    if not patch:
        first_tokens = 10**12
    return patch, -first_tokens


def _branch_events_from_dirs(trace_dirs: list[str | Path]) -> dict[str, dict[str, list[Event]]]:
    by_instance: dict[str, dict[str, list[Event]]] = defaultdict(lambda: defaultdict(list))
    for trace_dir in trace_dirs:
        for trace in load_trace_dir(trace_dir):
            source = Path(str(trace.metadata.get("source", trace.metadata.get("source_trace_dir", trace_dir)))).stem
            for ev in trace.events:
                if ev.branch_id == "root" or ev.node_type not in {"llm", "tool", "verifier"}:
                    continue
                branch_key = f"{ev.branch_id}:{source}"
                by_instance[ev.instance_id][branch_key].append(ev)
    for branches in by_instance.values():
        for key, events in branches.items():
            branches[key] = sorted(events, key=lambda e: (e.step_id, _event_time(e), e.node_id))
    return by_instance


def pabb_weights(path: str | Path = "configs/pabb_weights.yaml") -> dict[str, float]:
    data = read_yaml(path) if Path(path).exists() else {}
    raw = data.get("pabb", data)
    return {
        "w_patch": float(raw.get("patch_nonempty", 4.0)),
        "w_test": float(raw.get("pytest_command_seen", 1.5)),
        "w_return": float(raw.get("tool_returncode_success_or_progress", 0.8)),
        "w_file": float(raw.get("file_modification_seen", 1.0)),
        "w_duplicate": float(raw.get("duplicate_patch_hash", 2.0)),
        "w_error": float(raw.get("repeated_failure", 1.0)),
        "w_token": float(raw.get("normalized_token_cost", 1.0)),
        "w_stale": float(raw.get("no_file_modification", 1.0)),
        "w_age": float(raw.get("age_boost", 0.05)),
    }


def online_utility(st: BranchOnlineState, weights: dict[str, float], max_tokens: int) -> float:
    token_cost = st.llm_tokens_seen / max(1, max_tokens)
    return (
        weights["w_patch"] * float(st.patch_candidate_seen)
        + weights["w_test"] * float(st.pytest_seen)
        + weights["w_return"] * float(st.latest_returncode == 0)
        + weights["w_file"] * float(st.file_modification_seen)
        + 0.3 * min(1.0, st.modified_files_count / 3.0)
        + 0.2 * min(1.0, st.git_diff_stat_bytes / 512.0)
        - weights["w_duplicate"] * float(st.duplicate_patch_seen_so_far)
        - weights["w_error"] * st.error_count_seen
        - weights["w_token"] * token_cost
        - weights["w_stale"] * st.no_progress_steps
        + weights["w_age"] * st.age
    )


def choose_next_branches(
    states: dict[str, BranchOnlineState],
    policy: str,
    max_active: int,
    weights: dict[str, float] | None = None,
    max_tokens: int = 1,
) -> list[BranchOnlineState]:
    ready = [st for st in states.values() if st.alive and not st.done()]
    if not ready:
        return []
    if policy == "launch_all":
        return sorted(ready, key=lambda st: st.branch_id)
    if policy == "fcfs_budget":
        return sorted(ready, key=lambda st: st.branch_id)[:max_active]
    if policy == "pabb_oracle_upper_bound":
        return sorted(ready, key=lambda st: _future_patch_value(st.events[st.current_step_index :]), reverse=True)[:max_active]
    weights = weights or pabb_weights()
    return sorted(ready, key=lambda st: (-online_utility(st, weights, max_tokens), st.branch_id))[:max_active]


def update_branch_state(
    st: BranchOnlineState,
    ev: Event,
    lm: LatencyModel,
    seen_patch_hashes: set[str],
    now_step: int = 0,
) -> tuple[bool, str]:
    st.current_step_index += 1
    made_progress = False
    if ev.node_type in {"llm", "tool"}:
        st.steps_executed += 1
    if ev.node_type == "llm":
        st.llm_tokens_seen += ev.input_tokens + ev.output_tokens
        st.model_side_time += lm.predict_prefill(ev.input_tokens) + lm.predict_decode(ev.context_length or ev.input_tokens, ev.output_tokens)
        made_progress = ev.output_tokens > 0
    elif ev.node_type == "tool":
        command = ev.command or ""
        st.tool_outputs_seen += 1
        st.observation_tokens += ev.observation_tokens or 0
        st.pytest_seen = st.pytest_seen or _contains_test_command(command)
        st.latest_returncode = ev.exit_code
        st.error_count_seen += _error_count(command)
        snapshot_mod = bool(getattr(ev, "file_modification_seen", False))
        st.patch_snapshot_available = st.patch_snapshot_available or bool(getattr(ev, "patch_snapshot_available", False))
        st.modified_files_count = max(st.modified_files_count, int(getattr(ev, "modified_files_count", 0) or 0))
        st.untracked_files_count = max(st.untracked_files_count, int(getattr(ev, "untracked_files_count", 0) or 0))
        st.git_diff_stat_bytes = max(st.git_diff_stat_bytes, int(getattr(ev, "git_diff_stat_bytes", 0) or 0))
        st.file_modification_seen = st.file_modification_seen or snapshot_mod or (_looks_like_file_write(command) and ev.exit_code in {0, None})
        st.tool_time += float(ev.tool_latency if ev.tool_latency is not None else ev.latency or 0.0)
        made_progress = st.latest_returncode == 0 or st.file_modification_seen or st.pytest_seen
    elif ev.node_type == "verifier":
        if ev.verifier_result in {"pass", "fail"}:
            st.official_verifier_seen = ev.verifier_result
            made_progress = made_progress or ev.verifier_result == "pass"

    has_patch, patch_hash = _event_patch(ev)
    if has_patch:
        st.patch_candidate_seen = True
        st.patch_hash_seen = patch_hash
        st.duplicate_patch_seen_so_far = bool(patch_hash and patch_hash in seen_patch_hashes)
        if patch_hash:
            seen_patch_hashes.add(patch_hash)
        made_progress = True

    if made_progress:
        st.no_progress_steps = 0
        st.last_progress_time = now_step
    else:
        st.no_progress_steps += 1
    return has_patch, patch_hash


def _max_branch_tokens(branches: dict[str, list[Event]]) -> int:
    return max([sum(e.input_tokens + e.output_tokens for e in events if e.node_type == "llm") for events in branches.values()] or [1])


def _run_instance_policy(
    instance_id: str,
    branches: dict[str, list[Event]],
    policy: str,
    max_active: int,
    max_steps: int,
    max_tokens_per_instance: int,
    lm: LatencyModel,
    weights: dict[str, float],
) -> dict[str, Any]:
    states = {branch_id: BranchOnlineState(branch_id, events) for branch_id, events in sorted(branches.items())}
    max_tokens = _max_branch_tokens(branches)
    seen_patch_hashes: set[str] = set()
    first_patch: tuple[float, int, float] | None = None
    step = 0
    total_tokens = 0
    while True:
        for st in states.values():
            st.age += 1
            if st.steps_executed >= max_steps:
                st.alive = False
            if policy == "pabb_online" and st.duplicate_patch_seen_so_far:
                st.pruned = True
                st.alive = False
        if max_tokens_per_instance and total_tokens >= max_tokens_per_instance:
            break
        selected = choose_next_branches(states, policy, max_active, weights, max_tokens)
        if not selected:
            break
        for st in selected:
            if st.steps_executed >= max_steps:
                st.alive = False
                continue
            ev = st.next_event()
            if ev is None:
                st.alive = False
                continue
            before = sum(s.model_side_time + s.tool_time for s in states.values())
            has_patch, _ = update_branch_state(st, ev, lm, seen_patch_hashes, step)
            total_tokens = sum(s.llm_tokens_seen for s in states.values())
            after = sum(s.model_side_time + s.tool_time for s in states.values())
            if has_patch and first_patch is None:
                first_patch = (after, total_tokens, after)
            if max_tokens_per_instance and total_tokens >= max_tokens_per_instance:
                break
            if after == before and not has_patch:
                st.no_progress_steps += 0
        step += 1
        if first_patch is not None:
            break

    patch_hashes = [st.patch_hash_seen for st in states.values() if st.patch_hash_seen]
    duplicate_rate = (len(patch_hashes) - len(set(patch_hashes))) / max(1, len(patch_hashes))
    official = next((st.official_verifier_seen for st in states.values() if st.official_verifier_seen in {"pass", "fail"}), "unknown")
    return {
        "instance_id": instance_id,
        "policy": policy,
        "max_active_branches": max_active,
        "max_steps_per_branch": max_steps,
        "time_to_first_nonempty_patch": "" if first_patch is None else first_patch[0],
        "tokens_to_first_nonempty_patch": "" if first_patch is None else first_patch[1],
        "cost_to_first_nonempty_patch": "" if first_patch is None else first_patch[2],
        "branches_pruned": sum(1 for st in states.values() if st.pruned),
        "duplicate_patch_rate": duplicate_rate,
        "official_success_if_available": official,
        "verifier_unknown_count": sum(1 for st in states.values() if st.official_verifier_seen == "unknown"),
        "total_tokens_used": total_tokens,
        "total_tool_time": sum(st.tool_time for st in states.values()),
        "model_side_time": sum(st.model_side_time for st in states.values()),
        "pabb_online_gain_vs_fcfs": "",
        "oracle_gap": "",
    }


def _as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fill_gains(rows: list[dict[str, Any]]) -> None:
    fcfs = {
        (r["instance_id"], r["max_active_branches"], r["max_steps_per_branch"]): r
        for r in rows
        if r["policy"] == "fcfs_budget"
    }
    oracle = {
        (r["instance_id"], r["max_active_branches"], r["max_steps_per_branch"]): r
        for r in rows
        if r["policy"] == "pabb_oracle_upper_bound"
    }
    for row in rows:
        key = (row["instance_id"], row["max_active_branches"], row["max_steps_per_branch"])
        val = _as_float(row.get("cost_to_first_nonempty_patch"))
        base = _as_float(fcfs.get(key, {}).get("cost_to_first_nonempty_patch"))
        upper = _as_float(oracle.get(key, {}).get("cost_to_first_nonempty_patch"))
        if val is not None and base is not None and base > 0 and row["policy"] == "pabb_online":
            row["pabb_online_gain_vs_fcfs"] = (base - val) / base
        if val is not None and upper is not None and val > 0:
            row["oracle_gap"] = max(0.0, (val - upper) / val)


def run_pabb_online_v3(
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    weights_yaml: str | Path = "configs/pabb_weights.yaml",
    out_csv: str | Path = "data/results/pabb_online_branch_budget_pr4_v3.csv",
    plot_out: str | Path = "data/plots/pabb_online_branch_budget_pr4_v3.pdf",
    max_tokens_per_instance: int = 0,
) -> list[dict[str, Any]]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    lm = LatencyModel.load(model_json)
    weights = pabb_weights(weights_yaml)
    by_instance = _branch_events_from_dirs(trace_dirs)
    rows: list[dict[str, Any]] = []
    for instance_id, branches in sorted(by_instance.items()):
        for policy in ONLINE_POLICIES:
            for max_active in [1, 2, 4]:
                for max_steps in [5, 10, 15]:
                    rows.append(
                        _run_instance_policy(
                            instance_id,
                            branches,
                            policy,
                            max_active,
                            max_steps,
                            max_tokens_per_instance,
                            lm,
                            weights,
                        )
                    )
    _fill_gains(rows)
    write_csv(out_csv, rows)
    plot_pabb_online_v3(rows, plot_out)
    return rows


def _run_instance_snapshot_policy(
    instance_id: str,
    branches: dict[str, list[Event]],
    policy: str,
    max_active: int,
    max_steps: int,
    lm: LatencyModel,
    weights: dict[str, float],
) -> dict[str, Any]:
    mapped_policy = "pabb_online" if policy in {"pabb_online_v3", "pabb_snapshot_online"} else policy
    states = {branch_id: BranchOnlineState(branch_id, events) for branch_id, events in sorted(branches.items())}
    max_tokens = _max_branch_tokens(branches)
    seen_patch_hashes: set[str] = set()
    first_patch: tuple[float, int, float] | None = None
    first_file: tuple[float, int] | None = None
    step = 0
    while True:
        for st in states.values():
            st.age += 1
            if st.steps_executed >= max_steps:
                st.alive = False
            if mapped_policy == "pabb_online" and st.duplicate_patch_seen_so_far:
                st.pruned = True
                st.alive = False
        selected = choose_next_branches(states, mapped_policy, max_active, weights, max_tokens)
        if not selected:
            break
        for st in selected:
            if st.steps_executed >= max_steps:
                st.alive = False
                continue
            ev = st.next_event()
            if ev is None:
                st.alive = False
                continue
            has_patch, _ = update_branch_state(st, ev, lm, seen_patch_hashes, step)
            total_cost = sum(s.model_side_time + s.tool_time for s in states.values())
            total_tokens = sum(s.llm_tokens_seen for s in states.values())
            if st.file_modification_seen and first_file is None:
                first_file = (total_cost, total_tokens)
            if has_patch and first_patch is None:
                first_patch = (total_cost, total_tokens, total_cost)
        step += 1
        if first_patch is not None:
            break
    selected_states = list(states.values())
    official = next((s.official_verifier_seen for s in selected_states if s.official_verifier_seen in {"pass", "fail"}), "unknown")
    return {
        "instance_id": instance_id,
        "policy": policy,
        "max_active_branches": max_active,
        "max_steps_per_branch": max_steps,
        "time_to_first_file_modification": "" if first_file is None else first_file[0],
        "time_to_first_nonempty_patch": "" if first_patch is None else first_patch[0],
        "tokens_to_first_file_modification": "" if first_file is None else first_file[1],
        "tokens_to_first_nonempty_patch": "" if first_patch is None else first_patch[1],
        "cost_to_patch": "" if first_patch is None else first_patch[2],
        "branches_pruned": sum(1 for s in selected_states if s.pruned),
        "official_success_if_available": official,
        "snapshot_events_available": sum(1 for s in selected_states for e in s.events if getattr(e, "patch_snapshot_available", False)),
        "file_modification_seen": sum(1 for s in selected_states if s.file_modification_seen),
        "oracle_gap": "",
        "snapshot_gain_vs_fcfs": "",
        "snapshot_gain_vs_pabb_v3": "",
    }


def run_pabb_snapshot_online(
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    weights_yaml: str | Path = "configs/pabb_weights.yaml",
    out_csv: str | Path = "data/results/pabb_snapshot_online_pr4_v4.csv",
    plot_out: str | Path = "data/plots/pabb_snapshot_online_pr4_v4.pdf",
) -> list[dict[str, Any]]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    lm = LatencyModel.load(model_json)
    weights = pabb_weights(weights_yaml)
    by_instance = _branch_events_from_dirs(trace_dirs)
    rows: list[dict[str, Any]] = []
    for instance_id, branches in sorted(by_instance.items()):
        for policy in SNAPSHOT_POLICIES:
            for max_active in [1, 2, 4]:
                for max_steps in [5, 10, 15]:
                    rows.append(_run_instance_snapshot_policy(instance_id, branches, policy, max_active, max_steps, lm, weights))
    _fill_snapshot_gains(rows)
    write_csv(out_csv, rows)
    plot_pabb_snapshot(rows, plot_out)
    return rows


def _fill_snapshot_gains(rows: list[dict[str, Any]]) -> None:
    by_key: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(row["instance_id"], int(row["max_active_branches"]), int(row["max_steps_per_branch"]))][row["policy"]] = row
    for group in by_key.values():
        fcfs = group.get("fcfs_budget")
        pabb = group.get("pabb_online_v3")
        snap = group.get("pabb_snapshot_online")
        oracle = group.get("pabb_oracle_upper_bound")
        if snap:
            snap_cost = _as_float(snap.get("cost_to_patch"))
            fcfs_cost = _as_float(fcfs.get("cost_to_patch")) if fcfs else None
            pabb_cost = _as_float(pabb.get("cost_to_patch")) if pabb else None
            oracle_cost = _as_float(oracle.get("cost_to_patch")) if oracle else None
            if snap_cost is not None and fcfs_cost is not None and fcfs_cost > 0:
                snap["snapshot_gain_vs_fcfs"] = (fcfs_cost - snap_cost) / fcfs_cost
            if snap_cost is not None and pabb_cost is not None and pabb_cost > 0:
                snap["snapshot_gain_vs_pabb_v3"] = (pabb_cost - snap_cost) / pabb_cost
            if snap_cost is not None and oracle_cost is not None and snap_cost > 0:
                snap["oracle_gap"] = max(0.0, (snap_cost - oracle_cost) / snap_cost)


def plot_pabb_snapshot(rows: list[dict[str, Any]], out: str | Path) -> None:
    ensure_dir(Path(out).parent)
    vals: dict[str, list[float]] = defaultdict(list)
    file_vals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        c = _as_float(row.get("cost_to_patch"))
        f = _as_float(row.get("time_to_first_file_modification"))
        if c is not None:
            vals[row["policy"]].append(c)
        if f is not None:
            file_vals[row["policy"]].append(f)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].bar(SNAPSHOT_POLICIES, [sum(vals[p]) / len(vals[p]) if vals[p] else 0.0 for p in SNAPSHOT_POLICIES])
    axes[0].set_ylabel("cost to non-empty patch")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(SNAPSHOT_POLICIES, [sum(file_vals[p]) / len(file_vals[p]) if file_vals[p] else 0.0 for p in SNAPSHOT_POLICIES])
    axes[1].set_ylabel("time to file modification")
    axes[1].tick_params(axis="x", rotation=25)
    fig.savefig(out)
    plt.close(fig)


def plot_pabb_online_v3(rows: list[dict[str, Any]], out: str | Path) -> None:
    ensure_dir(Path(out).parent)
    policies = ONLINE_POLICIES
    cost: dict[str, list[float]] = defaultdict(list)
    tokens: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        c = _as_float(row.get("cost_to_first_nonempty_patch"))
        t = _as_float(row.get("tokens_to_first_nonempty_patch"))
        if c is not None:
            cost[row["policy"]].append(c)
        if t is not None:
            tokens[row["policy"]].append(t)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].bar(policies, [sum(cost[p]) / len(cost[p]) if cost[p] else 0.0 for p in policies])
    axes[0].set_ylabel("online cost to first non-empty patch")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(policies, [sum(tokens[p]) / len(tokens[p]) if tokens[p] else 0.0 for p in policies])
    axes[1].set_ylabel("online tokens to first non-empty patch")
    axes[1].tick_params(axis="x", rotation=25)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dirs", default="data/traces/mini_swe_lite10_r4_timed,data/traces/mini_swe_lite5_patchcap_verified")
    ap.add_argument("--model-json", default="data/profiles/h100_latency_model_pr2_v2.json")
    ap.add_argument("--weights-yaml", default="configs/pabb_weights.yaml")
    ap.add_argument("--out", default="data/results/pabb_online_branch_budget_pr4_v3.csv")
    ap.add_argument("--plot-out", default="data/plots/pabb_online_branch_budget_pr4_v3.pdf")
    ap.add_argument("--max-tokens-per-instance", type=int, default=0)
    ap.add_argument("--snapshot", action="store_true")
    args = ap.parse_args()
    if args.snapshot:
        rows = run_pabb_snapshot_online(
            [x for x in args.trace_dirs.split(",") if x.strip()],
            args.model_json,
            args.weights_yaml,
            args.out,
            args.plot_out,
        )
    else:
        rows = run_pabb_online_v3(
            [x for x in args.trace_dirs.split(",") if x.strip()],
            args.model_json,
            args.weights_yaml,
            args.out,
            args.plot_out,
            args.max_tokens_per_instance,
        )
    print(json.dumps({"rows": len(rows), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
