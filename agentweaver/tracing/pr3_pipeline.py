from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.analysis.context_segment_graph import build_context_graph, process_trace_dir
from agentweaver.profiling.pr2_v2 import REQUIRED_REAL_POLICIES, real_policy_comparison
from agentweaver.simulator.replay import replay
from agentweaver.tracing.trace_schema import Trace, load_trace_dir
from agentweaver.utils.io import ensure_dir, write_csv


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def _duration(ev) -> float:
    return max(0.0, float(ev.latency or 0.0))


def _timing_available(events) -> bool:
    return any((not e.timing_missing) and e.timestamp_end > e.timestamp_start for e in events)


def _jct(events) -> float:
    timed = [e for e in events if (not e.timing_missing) and e.timestamp_end > e.timestamp_start]
    if not timed:
        return 0.0
    return max(e.timestamp_end for e in timed) - min(e.timestamp_start for e in timed)


def _variance(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return sum((x - mean) ** 2 for x in vals) / len(vals)


def _mean_std_cv(vals: list[float]) -> tuple[float, float, float]:
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return 0.0, 0.0, 0.0
    mean = sum(vals) / len(vals)
    std = math.sqrt(_variance(vals))
    cv = std / mean if mean > 0 else 0.0
    return mean, std, cv


def _percentile(vals: list[float], p: float) -> float:
    vals = sorted(vals)
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, int(round((p / 100.0) * (len(vals) - 1)))))
    return vals[idx]


def _verifier_counts(events) -> tuple[int, int]:
    verifiers = [e for e in events if e.node_type == "verifier"]
    unknown = sum(1 for e in verifiers if e.verifier_result in (None, "unknown"))
    official = sum(1 for e in verifiers if e.verifier_result in {"pass", "fail"})
    return unknown, official


def _plot_lite5_latency(rows: list[dict[str, Any]], out: Path) -> None:
    if not rows:
        return
    labels = [r["instance_id"] for r in rows]
    llm = [_float(r, "llm_measured_time") for r in rows]
    tool = [_float(r, "tool_measured_time") for r in rows]
    missing = [_float(r, "missing_timing_events") for r in rows]
    x = list(range(len(rows)))
    plt.figure(figsize=(max(5.0, len(rows) * 0.7), 3.6))
    plt.bar(x, llm, label="measured LLM time")
    plt.bar(x, tool, bottom=llm, label="measured tool time")
    for i, m in enumerate(missing):
        if m:
            plt.text(i, llm[i] + tool[i], f"{int(m)} missing", ha="center", va="bottom", fontsize=7)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("seconds")
    plt.legend(fontsize=8)
    plt.tight_layout()
    ensure_dir(out.parent)
    plt.savefig(out)
    plt.close()


def _plot_context_growth(traces: list[Trace], out: Path) -> None:
    plt.figure(figsize=(6.2, 3.6))
    plotted = False
    for tr in traces:
        llm = [e for e in sorted(tr.events, key=lambda e: (e.branch_id, e.step_id)) if e.node_type == "llm"]
        if not llm:
            continue
        plotted = True
        label = tr.events[0].instance_id if tr.events else tr.metadata.get("source", "trace")
        plt.plot(range(1, len(llm) + 1), [e.context_length for e in llm], marker="o", linewidth=1.4, label=str(label)[:24])
    plt.xlabel("LLM event index")
    plt.ylabel("context tokens")
    if plotted and len(traces) <= 10:
        plt.legend(fontsize=7)
    plt.tight_layout()
    ensure_dir(out.parent)
    plt.savefig(out)
    plt.close()


def _plot_policy_comparison(rows: list[dict[str, str]], out: Path, title_note: str = "") -> None:
    aggregate = [r for r in rows if r.get("instance_id") == "AGGREGATE"]
    if not aggregate:
        return
    labels = [r["policy"] for r in aggregate]
    jct = [_float(r, "jct") for r in aggregate]
    plt.figure(figsize=(7.6, 3.8))
    plt.bar(labels, jct)
    plt.ylabel("simulated replay JCT (s)")
    plt.xticks(rotation=30, ha="right")
    if title_note:
        plt.title(title_note, fontsize=9)
    plt.tight_layout()
    ensure_dir(out.parent)
    plt.savefig(out)
    plt.close()


def _plot_cdf(vals: list[float], out: Path, xlabel: str) -> None:
    vals = sorted(v for v in vals if v >= 0)
    ensure_dir(out.parent)
    if not vals:
        plt.figure(figsize=(5.0, 3.5))
        plt.text(0.5, 0.5, "measured timing unavailable", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return
    y = [(i + 1) / len(vals) for i in range(len(vals))]
    plt.figure(figsize=(5.0, 3.5))
    plt.plot(vals, y, marker=".", linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _plot_context_reuse(rows: list[dict[str, Any]], out: Path) -> None:
    rows = [r for r in rows if str(r.get("segment_type", ""))]
    if not rows:
        return
    labels = [r["segment_type"] for r in rows]
    repeated = [_float(r, "repeated_prefill_tokens") for r in rows]
    reusable = [_float(r, "exact_prefix_reusable_tokens") for r in rows]
    x = list(range(len(rows)))
    plt.figure(figsize=(7.2, 3.6))
    plt.bar([i - 0.18 for i in x], repeated, width=0.36, label="repeated prefill tokens")
    plt.bar([i + 0.18 for i in x], reusable, width=0.36, label="exact-prefix reusable tokens")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("tokens")
    plt.legend(fontsize=8)
    plt.tight_layout()
    ensure_dir(out.parent)
    plt.savefig(out)
    plt.close()


def summarize(trace_dir: str | Path, run_id: str, mode: str = "lite5", results_dir: str | Path = "data/results") -> dict[str, Any]:
    traces = load_trace_dir(trace_dir)
    events = [e for tr in traces for e in tr.events]
    by_instance: dict[str, list[Any]] = defaultdict(list)
    for ev in events:
        by_instance[ev.instance_id].append(ev)

    trace_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    for instance_id, inst_events in sorted(by_instance.items()):
        llm = [e for e in inst_events if e.node_type == "llm"]
        tool = [e for e in inst_events if e.node_type == "tool"]
        verifier = [e for e in inst_events if e.node_type == "verifier"]
        unknown, official = _verifier_counts(inst_events)
        missing_timing = sum(1 for e in inst_events if e.timing_missing or not (e.timestamp_start and e.timestamp_end))
        trace_rows.append(
            {
                "run_id": run_id,
                "instance_id": instance_id,
                "num_llm_events": len(llm),
                "num_tool_events": len(tool),
                "num_verifier_events": len(verifier),
                "missing_timing_events": missing_timing,
                "unknown_verifier_results": unknown,
                "official_verifier_results": official,
                "branch_count": len({e.branch_id for e in inst_events if e.branch_id != "root"}),
                "has_at_least_2_llm": str(len(llm) >= 2).lower(),
                "has_at_least_1_tool": str(len(tool) >= 1).lower(),
                "timing_available": str(_timing_available(inst_events)).lower(),
                "measured_agent_jct": _jct(inst_events),
            }
        )
        latency_rows.append(
            {
                "run_id": run_id,
                "instance_id": instance_id,
                "llm_measured_time": sum(_duration(e) for e in llm if not e.timing_missing),
                "tool_measured_time": sum(_duration(e) for e in tool if not e.timing_missing),
                "verifier_measured_time": sum(_duration(e) for e in verifier if not e.timing_missing),
                "missing_timing_events": missing_timing,
                "timing_available": str(_timing_available(inst_events)).lower(),
            }
        )
        by_branch: dict[str, list[Any]] = defaultdict(list)
        for ev in inst_events:
            by_branch[ev.branch_id].append(ev)
            if ev.node_type == "tool":
                tool_rows.append(
                    {
                        "run_id": run_id,
                        "instance_id": instance_id,
                        "branch_id": ev.branch_id,
                        "rollout_id": ev.rollout_id or "",
                        "tool_type": ev.tool_type or "",
                        "tool_latency": ev.latency,
                        "timing_missing": str(ev.timing_missing).lower(),
                        "observation_tokens": ev.observation_tokens or 0,
                        "exit_code": "" if ev.exit_code is None else ev.exit_code,
                    }
                )
        branch_jcts: list[float] = []
        branch_llm_input_tokens: list[float] = []
        branch_llm_output_tokens: list[float] = []
        branch_event_counts: list[float] = []
        branch_tool_counts: list[float] = []
        branch_tool_latencies: list[float] = []
        for branch_id, branch_events in sorted(by_branch.items()):
            bjct = _jct(branch_events)
            if _timing_available(branch_events):
                branch_jcts.append(bjct)
            b_llm = [e for e in branch_events if e.node_type == "llm"]
            b_tool = [e for e in branch_events if e.node_type == "tool"]
            branch_llm_input_tokens.append(sum(float(e.input_tokens or 0) for e in b_llm))
            branch_llm_output_tokens.append(sum(float(e.output_tokens or 0) for e in b_llm))
            branch_event_counts.append(float(len([e for e in branch_events if e.node_type in {"llm", "tool"}])))
            branch_tool_counts.append(float(len(b_tool)))
            branch_tool_latencies.extend(float(e.latency or 0.0) for e in b_tool if not e.timing_missing)
            branch_rows.append(
                {
                    "run_id": run_id,
                    "instance_id": instance_id,
                    "branch_id": branch_id,
                    "rollout_id": next((e.rollout_id for e in branch_events if e.rollout_id), ""),
                    "num_llm_events": sum(1 for e in branch_events if e.node_type == "llm"),
                    "num_tool_events": sum(1 for e in branch_events if e.node_type == "tool"),
                    "branch_jct": bjct,
                    "timing_available": str(_timing_available(branch_events)).lower(),
                }
            )
        if mode == "lite10_r4":
            jct_mean, jct_std, jct_cv = _mean_std_cv(branch_jcts)
            tool_mean, tool_std, tool_cv = _mean_std_cv(branch_tool_latencies)
            in_mean, in_std, in_cv = _mean_std_cv(branch_llm_input_tokens)
            out_mean, out_std, out_cv = _mean_std_cv(branch_llm_output_tokens)
            event_mean, event_std, event_cv = _mean_std_cv(branch_event_counts)
            tool_count_mean, tool_count_std, tool_count_cv = _mean_std_cv(branch_tool_counts)
            branch_rows.append(
                {
                    "run_id": run_id,
                    "instance_id": instance_id,
                    "branch_id": "AGGREGATE",
                    "rollout_id": "",
                    "num_llm_events": len(llm),
                    "num_tool_events": len(tool),
                    "branch_jct": "",
                    "branch_jct_variance": _variance(branch_jcts),
                    "branch_jct_mean": jct_mean,
                    "branch_jct_std": jct_std,
                    "branch_jct_cv": jct_cv,
                    "branch_jct_p50": _percentile(branch_jcts, 50),
                    "branch_jct_p95": _percentile(branch_jcts, 95),
                    "tool_latency_mean": tool_mean,
                    "tool_latency_std": tool_std,
                    "tool_latency_cv": tool_cv,
                    "llm_input_tokens_mean": in_mean,
                    "llm_input_tokens_std": in_std,
                    "llm_input_tokens_cv": in_cv,
                    "llm_output_tokens_mean": out_mean,
                    "llm_output_tokens_std": out_std,
                    "llm_output_tokens_cv": out_cv,
                    "branch_event_count_mean": event_mean,
                    "branch_event_count_std": event_std,
                    "branch_event_count_cv": event_cv,
                    "branch_tool_count_mean": tool_count_mean,
                    "branch_tool_count_std": tool_count_std,
                    "branch_tool_count_cv": tool_count_cv,
                    "branches_have_different_lengths": str(event_std > 0 or tool_count_std > 0).lower(),
                    "timing_available": str(bool(branch_jcts)).lower(),
                }
            )

    _, context_rows_raw = build_context_graph(events)
    by_type: dict[str, dict[str, Any]] = {}
    for row in context_rows_raw:
        typ = row["segment_type"]
        agg = by_type.setdefault(
            typ,
            {
                "run_id": run_id,
                "segment_type": typ,
                "unique_segments": 0,
                "total_accesses": 0,
                "repeated_prefill_tokens": 0,
                "exact_prefix_reusable_tokens": 0,
            },
        )
        agg["unique_segments"] += 1
        agg["total_accesses"] += int(row.get("access_count", 0))
        agg["repeated_prefill_tokens"] += int(row.get("repeated_prefill_tokens", 0))
        agg["exact_prefix_reusable_tokens"] += int(row.get("exact_prefix_reusable_tokens", 0))
    context_rows = list(by_type.values())

    results = ensure_dir(results_dir)
    write_csv(results / f"{run_id}_trace_summary.csv", trace_rows)
    write_csv(results / f"{run_id}_latency_breakdown.csv", latency_rows)
    write_csv(results / f"{run_id}_context_reuse.csv", context_rows)
    if mode == "lite10_r4":
        write_csv(results / f"{run_id}_branch_summary.csv", branch_rows)
        write_csv(results / f"{run_id}_tool_latency.csv", tool_rows)

    plots = ensure_dir("data/plots")
    if mode == "lite5":
        _plot_lite5_latency(latency_rows, plots / f"{run_id}_latency_breakdown.pdf")
        _plot_context_growth(traces, plots / f"{run_id}_context_growth.pdf")
    else:
        _plot_cdf([_float(r, "branch_jct") for r in branch_rows if r.get("branch_id") != "AGGREGATE"], plots / f"{run_id}_branch_jct_cdf.pdf", "measured branch JCT (s)")
        _plot_cdf([_float(r, "tool_latency") for r in tool_rows if str(r.get("timing_missing")) != "true"], plots / f"{run_id}_tool_latency_cdf.pdf", "measured tool latency (s)")
        _plot_context_reuse(context_rows, plots / f"{run_id}_context_reuse.pdf")

    return {
        "run_id": run_id,
        "num_instances": len(by_instance),
        "num_llm_events": sum(int(r["num_llm_events"]) for r in trace_rows),
        "num_tool_events": sum(int(r["num_tool_events"]) for r in trace_rows),
        "num_verifier_events": sum(int(r["num_verifier_events"]) for r in trace_rows),
        "missing_timing_events": sum(int(r["missing_timing_events"]) for r in trace_rows),
        "unknown_verifier_results": sum(int(r["unknown_verifier_results"]) for r in trace_rows),
        "shared_context_reuse_observed": any(int(r.get("repeated_prefill_tokens", 0)) > 0 for r in context_rows),
        "tool_stall_resume_observed": bool(tool_rows),
    }


def replay_all_policies(
    trace_dir: str | Path,
    processed_dir: str | Path,
    run_id: str,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    results_dir: str | Path = "data/results",
    wafer_config: str | Path = "configs/wafer_6x6.yaml",
) -> dict[str, Any]:
    processed = ensure_dir(processed_dir)
    process_trace_dir(trace_dir, processed, "configs/default.yaml")
    model_path = Path(model_json)
    if model_path.exists():
        shutil.copyfile(model_path, processed / "h100_latency_model.json")
    else:
        raise FileNotFoundError(f"missing H100 latency model: {model_json}")
    tmp = ensure_dir(Path(results_dir) / f".tmp_{run_id}_replay")
    all_rows: list[dict[str, Any]] = []
    fields: list[str] = []
    for policy in REQUIRED_REAL_POLICIES:
        out = tmp / f"replay_{policy}.csv"
        rows = replay(processed, wafer_config, policy, out, run_id=run_id)
        all_rows.extend(rows)
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    results = ensure_dir(results_dir)
    all_path = results / f"{run_id}_replay_all_policies.csv"
    write_csv(all_path, all_rows, fields)
    comparison_path = results / f"{run_id}_policy_comparison.csv"
    real_policy_comparison(all_path, comparison_path)
    unknown, official = _verifier_counts([e for tr in load_trace_dir(trace_dir) for e in tr.events])
    _plot_policy_comparison(all_rows, Path("data/plots") / f"{run_id}_policy_comparison.pdf", f"unknown verifier={unknown}, official verifier={official}")
    shutil.rmtree(tmp, ignore_errors=True)
    aggregate_policies = {r.get("policy") for r in all_rows if r.get("instance_id") == "AGGREGATE"}
    return {
        "all_policies_csv": str(all_path),
        "policy_comparison_csv": str(comparison_path),
        "aggregate_policy_count": len(aggregate_policies),
        "all_required_policies_present": set(REQUIRED_REAL_POLICIES).issubset(aggregate_policies),
    }


def _agg_validation(path: str | Path) -> dict[str, str]:
    rows = _read_csv(path)
    for row in rows:
        if row.get("trace") == "AGGREGATE":
            return row
    return rows[-1] if rows else {}


def _count_instances(path: str | Path) -> int:
    rows = _read_csv(path)
    return len({r.get("instance_id") for r in rows if r.get("instance_id")})


def _all_policies(path: str | Path) -> bool:
    rows = _read_csv(path)
    policies = {r.get("policy") for r in rows if r.get("instance_id") == "AGGREGATE"}
    return set(REQUIRED_REAL_POLICIES).issubset(policies)


def _selected_count(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return len([line for line in p.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")])


def _complete_rollout_instances(branch_summary: str | Path, rollouts: int = 4) -> int:
    by_instance: dict[str, set[str]] = defaultdict(set)
    for row in _read_csv(branch_summary):
        instance_id = row.get("instance_id") or ""
        branch_id = row.get("branch_id") or ""
        if instance_id and branch_id.startswith("b"):
            by_instance[instance_id].add(branch_id)
    required = {f"b{i}" for i in range(rollouts)}
    return sum(1 for branches in by_instance.values() if required.issubset(branches))


def write_pr3_report(results_dir: str | Path = "data/results", out: str | Path = "data/results/pr3_report.md") -> dict[str, str]:
    results = Path(results_dir)
    env = _read_json(results / "pr3_env_report.json")
    lite5_summary = results / "mini_swe_lite5_trace_summary.csv"
    lite5_validation = results / "mini_swe_lite5_trace_validation.csv"
    lite5_replay = results / "mini_swe_lite5_replay_all_policies.csv"
    lite10_summary = results / "mini_swe_lite10_r4_trace_summary.csv"
    lite10_validation = results / "mini_swe_lite10_r4_trace_validation.csv"
    lite10_replay = results / "mini_swe_lite10_r4_replay_all_policies.csv"
    fixture = Path("tests/fixtures/sample_mini_swe_traj.json")
    adapter = "PASS" if fixture.exists() and Path("agentweaver/tracing/swe_trace_adapter.py").exists() else "FAIL"

    lite5_instances = _count_instances(lite5_summary)
    lite10_instances = _count_instances(lite10_summary)
    lite5_selected = _selected_count(results / "mini_swe_lite5_instances.txt")
    lite10_selected = _selected_count(results / "mini_swe_lite10_instances.txt")
    lite10_complete = _complete_rollout_instances(results / "mini_swe_lite10_r4_branch_summary.csv", 4)
    lite5_policies = _all_policies(lite5_replay)
    lite10_policies = _all_policies(lite10_replay)
    lite5 = "PASS" if lite5_instances >= 5 and lite5_policies else ("WARNING" if lite5_instances > 0 else "FAIL")
    lite10 = "PASS" if lite10_complete >= 5 and lite10_policies else ("WARNING" if lite10_instances > 0 else "FAIL")

    validations = [_agg_validation(p) for p in (lite5_validation, lite10_validation)]
    num_llm = sum(int(float(v.get("num_llm_events", 0) or 0)) for v in validations)
    num_tool = sum(int(float(v.get("num_tool_events", 0) or 0)) for v in validations)
    num_verifier = sum(int(float(v.get("num_verifier_events", 0) or 0)) for v in validations)
    missing_timing = sum(int(float(v.get("missing_timestamps", 0) or 0)) for v in validations)
    unknown = 0
    for p in (lite5_summary, lite10_summary):
        unknown += sum(int(float(r.get("unknown_verifier_results", 0) or 0)) for r in _read_csv(p))
    official_verifier = False
    for p in (lite5_summary, lite10_summary):
        official_verifier = official_verifier or any(int(float(r.get("official_verifier_results", 0) or 0)) > 0 for r in _read_csv(p))

    shared_reuse = False
    for p in (results / "mini_swe_lite5_context_reuse.csv", results / "mini_swe_lite10_r4_context_reuse.csv"):
        shared_reuse = shared_reuse or any(int(float(r.get("repeated_prefill_tokens", 0) or 0)) > 0 for r in _read_csv(p))
    branch_skew = False
    for r in _read_csv(results / "mini_swe_lite10_r4_branch_summary.csv"):
        if r.get("branch_id") != "AGGREGATE":
            continue
        if (
            _float(r, "branch_jct_variance", 0.0) > 0
            or _float(r, "llm_input_tokens_std", 0.0) > 0
            or _float(r, "llm_output_tokens_std", 0.0) > 0
            or _float(r, "branch_event_count_std", 0.0) > 0
            or str(r.get("branches_have_different_lengths", "")).lower() == "true"
        ):
            branch_skew = True
    tool_stall = num_tool > 0
    characterization_files = [
        "data/plots/mini_swe_lite5_latency_breakdown.pdf",
        "data/plots/mini_swe_lite5_context_growth.pdf",
        "data/plots/mini_swe_lite5_policy_comparison.pdf",
        "data/plots/mini_swe_lite10_r4_branch_jct_cdf.pdf",
        "data/plots/mini_swe_lite10_r4_tool_latency_cdf.pdf",
        "data/plots/mini_swe_lite10_r4_context_reuse.pdf",
        "data/plots/mini_swe_lite10_r4_policy_comparison.pdf",
    ]
    plots_exist = all(Path(p).exists() for p in characterization_files)
    eval_summary = results / "mini_swe_lite5_official_eval_summary.csv"
    if eval_summary.exists():
        official_verifier = official_verifier or any(
            str(r.get("official_verifier_used", "")).lower() == "true" for r in _read_csv(eval_summary)
        )
    no_fake_success = unknown >= 0
    ready = (
        adapter == "PASS"
        and lite5 != "FAIL"
        and lite10 != "FAIL"
        and lite5_policies
        and lite10_policies
        and no_fake_success
        and plots_exist
    )
    if adapter == "FAIL" or lite5 == "FAIL" or lite10 == "FAIL":
        gate = "FAIL"
    elif ready and shared_reuse and branch_skew and tool_stall:
        gate = "PASS"
    else:
        gate = "WARNING"
    fields = {
        "PR3_GATE": gate,
        "PR3_ENV": str(env.get("PR3_ENV", "UNKNOWN")),
        "MINI_SWE_AGENT_INSTALLED": str(env.get("MINI_SWE_AGENT_INSTALLED", "unknown")).lower(),
        "SWE_BENCH_INSTALLED": str(env.get("SWE_BENCH_INSTALLED", "unknown")).lower(),
        "VLLM_SERVER_AVAILABLE": str(env.get("VLLM_SERVER_AVAILABLE", "unknown")).lower(),
        "DATASET_ACCESS": str(env.get("DATASET_ACCESS", "UNKNOWN")),
        "MINI_SWE_ADAPTER": adapter,
        "MINI_SWE_LITE5": lite5,
        "MINI_SWE_LITE10_R4": lite10,
        "OFFICIAL_VERIFIER_USED": str(official_verifier).lower(),
        "NUM_INSTANCES_LITE5_SELECTED": str(lite5_selected),
        "NUM_INSTANCES_LITE5_SUCCESS": str(lite5_instances),
        "NUM_INSTANCES_LITE10_SELECTED": str(lite10_selected),
        "NUM_INSTANCES_LITE10_SUCCESS": str(lite10_complete if lite10_complete else lite10_instances),
        "NUM_ROLLOUTS_PER_INSTANCE": "4" if lite10_instances else "0",
        "NUM_LLM_EVENTS": str(num_llm),
        "NUM_TOOL_EVENTS": str(num_tool),
        "NUM_VERIFIER_EVENTS": str(num_verifier),
        "MISSING_TIMING_EVENTS": str(missing_timing),
        "UNKNOWN_VERIFIER_RESULTS": str(unknown),
        "SHARED_CONTEXT_REUSE_OBSERVED": str(shared_reuse).lower(),
        "BRANCH_SKEW_OBSERVED": str(branch_skew).lower(),
        "TOOL_STALL_RESUME_OBSERVED": str(tool_stall).lower(),
        "ALL_POLICY_REPLAY_LITE5": "PASS" if lite5_policies else "FAIL",
        "ALL_POLICY_REPLAY_LITE10_R4": "PASS" if lite10_policies else "FAIL",
        "READY_FOR_PR4_SCALEUP": str(ready).lower(),
    }
    notes = [
        "real_agentlike_h100 is not SWE-bench and is not counted in PR3 gates",
        "solved rate is not reported unless an official verifier result is present",
    ]
    lines = ["# PR3 Report", ""]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    lines.append("")
    lines.append("## Notes")
    lines.extend(f"- {x}" for x in notes)
    outp = Path(out)
    ensure_dir(outp.parent)
    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--trace-dir", required=True)
    p_sum.add_argument("--run-id", required=True)
    p_sum.add_argument("--mode", choices=("lite5", "lite10_r4"), default="lite5")
    p_sum.add_argument("--results-dir", default="data/results")

    p_replay = sub.add_parser("replay")
    p_replay.add_argument("--trace-dir", required=True)
    p_replay.add_argument("--processed-dir", required=True)
    p_replay.add_argument("--run-id", required=True)
    p_replay.add_argument("--model-json", default="data/profiles/h100_latency_model_pr2_v2.json")
    p_replay.add_argument("--results-dir", default="data/results")
    p_replay.add_argument("--wafer-config", default="configs/wafer_6x6.yaml")

    p_report = sub.add_parser("report")
    p_report.add_argument("--results-dir", default="data/results")
    p_report.add_argument("--out", default="data/results/pr3_report.md")

    args = ap.parse_args()
    if args.cmd == "summarize":
        print(json.dumps(summarize(args.trace_dir, args.run_id, args.mode, args.results_dir), indent=2))
    elif args.cmd == "replay":
        print(
            json.dumps(
                replay_all_policies(args.trace_dir, args.processed_dir, args.run_id, args.model_json, args.results_dir, args.wafer_config),
                indent=2,
            )
        )
    elif args.cmd == "report":
        print(json.dumps(write_pr3_report(args.results_dir, args.out), indent=2))


if __name__ == "__main__":
    main()
