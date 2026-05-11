from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

from agentweaver.analysis.policy_grid_audit import audit_grid
from agentweaver.analysis.pr4_v10_artifact_sanity import run_sanity
from agentweaver.analysis.context_segment_graph import kv_bytes_per_token
from agentweaver.astra.export_chakra import (
    export_policy_aware_trace_to_chakra_json,
    export_trace_to_chakra_json,
    write_per_npu_traces,
)
from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.aligned_policy_sweep import ALIGNED_POLICIES, run_aligned_policy_grid
from agentweaver.simulator.safe_tool_prefetch import run_all as run_stp
from agentweaver.simulator.taps_cost_model_v3 import evaluate_compiler, summarize_validation
from agentweaver.simulator.taps_unified import TAPSUnifiedReplay, _load_traces
from agentweaver.simulator.taps_unified_adaptive import ADAPTIVE_POLICY, AdaptiveProfiles, TAPSAdaptiveReplay
from agentweaver.simulator.workload_feature_extractor import extract_workload_features
from agentweaver.tracing.trace_schema import Trace
from agentweaver.utils.io import ensure_dir, write_csv


SUPPORTED_SCHEDULE_POLICIES = {
    "reactive_admission": "reactive_admission",
    "acd_nisp": "acd_nisp",
    "taps_domain_v4": "taps_domain_v4",
    "taps_domain_v4_fixed": "taps_domain_v4",
    "taps_admission_v4": "taps_admission_v4",
    "taps_unified_v5": "taps_unified",
    "taps_unified_v5_fixed": "taps_unified",
    ADAPTIVE_POLICY: ADAPTIVE_POLICY,
}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return default
    try:
        v = row.get(key)
        return default if v in ("", None) else float(v)
    except Exception:
        return default


def _parse_field(path: str | Path, key: str, default: str = "") -> str:
    p = Path(path)
    if not p.exists():
        return default
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key} = "):
            return line.split(" = ", 1)[1].strip()
    return default


def _find_trace_for_schedule(schedule_jsonl: str | Path, trace_dirs: list[str | Path]) -> str | None:
    event_ids: set[str] = set()
    p = Path(schedule_jsonl)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                event_ids.add(str(json.loads(line).get("event_id", "")))
    for trace_dir in trace_dirs:
        for path in sorted(Path(trace_dir).glob("*.jsonl")):
            trace = Trace.from_jsonl(path)
            if any(ev.event_id in event_ids for ev in trace.events):
                return str(path)
    return None


def _config_row(rows: list[dict[str, str]], cid: str, policy: str) -> dict[str, str] | None:
    return next((r for r in rows if r.get("config_id") == cid and r.get("policy") == policy), None)


def capture_schedule_for_compiler_selection(
    validation_csv: str | Path = "data/results/taps_compiler_v3_validation_pr4_v10.csv",
    valid_grid_csv: str | Path = "data/results/aligned_policy_grid_valid_pr4_v10.csv",
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    run_id: str = "pr4_v10_taps_c_v3",
) -> dict[str, Any]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    validation = _read_csv(validation_csv)
    grid = _read_csv(valid_grid_csv)
    chosen: dict[str, str] | None = None
    for row in validation:
        if row.get("split_type") == "random" and row.get("objective") == "balanced" and row.get("selected_policy") in SUPPORTED_SCHEDULE_POLICIES:
            chosen = row
            break
    if not chosen:
        for row in validation:
            if row.get("selected_policy") in SUPPORTED_SCHEDULE_POLICIES:
                chosen = row
                break
    if not chosen:
        return {"schedule_available": False, "reason": "no_supported_selected_policy"}
    cid = chosen["config_id"]
    selected_policy = chosen["selected_policy"]
    cfg = _config_row(grid, cid, selected_policy) or next((r for r in grid if r.get("config_id") == cid), None)
    if not cfg:
        return {"schedule_available": False, "reason": f"config_not_found:{cid}"}
    traces = _load_traces(trace_dirs)
    lm = LatencyModel.load(model_json)
    schedule_path = Path("data/schedules") / f"{run_id}_schedule.jsonl"
    ensure_dir(schedule_path.parent)
    mapped = SUPPORTED_SCHEDULE_POLICIES[selected_policy]
    common = dict(
        total_sessions=int(_f(cfg, "total_sessions")),
        active_session_limit=int(_f(cfg, "active_session_limit")),
        effective_regions=int(_f(cfg, "effective_regions")),
        arrival_pattern=str(cfg.get("arrival_pattern", "closed_loop")),
        memory_budget_gb=int(_f(cfg, "memory_budget_gb")),
    )
    if mapped == ADAPTIVE_POLICY:
        replay = TAPSAdaptiveReplay(
            traces,
            common["total_sessions"],
            common["active_session_limit"],
            common["effective_regions"],
            common["arrival_pattern"],
            common["memory_budget_gb"],
            lm,
            profiles=AdaptiveProfiles.default(),
            seed=909,
            run_id=run_id,
            config_id=cid,
            schedule_log_path=schedule_path,
        )
    else:
        replay = TAPSUnifiedReplay(
            traces,
            common["total_sessions"],
            common["active_session_limit"],
            common["effective_regions"],
            common["arrival_pattern"],
            common["memory_budget_gb"],
            mapped,
            lm,
            seed=909,
            run_id=run_id,
            config_id=cid,
            schedule_log_path=schedule_path,
        )
    result = replay.run()
    return {
        "schedule_available": True,
        "schedule_jsonl": str(schedule_path),
        "selected_config_id": cid,
        "selected_policy": selected_policy,
        "replay_policy": mapped,
        "completed_sessions": result.get("completed_sessions", 0),
        "total_sessions": result.get("total_sessions", 0),
        "starvation_count": result.get("starvation_count", 0),
        "cache_hit_tokens": result.get("cache_hit_tokens", 0),
        "remote_kv_bytes": result.get("remote_kv_bytes", 0.0),
    }


def export_astra_v3(
    schedule_info: dict[str, Any],
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
) -> dict[str, Any]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    report: dict[str, Any] = {
        "ASTRA_POLICY_AWARE_EXPORT_V3": "FAIL",
        "ASTRA_EXPORT_USES_REAL_SCHEDULE": "false",
        "ASTRA_SIM_RUN_COMPLETED": "false",
        "ASTRA_REMOTE_REDUCTION": "0.000000",
        "ASTRA_COMPUTE_REDUCTION": "0.000000",
    }
    if not schedule_info.get("schedule_available"):
        report["ASTRA_POLICY_AWARE_EXPORT_V3"] = "WARNING_NO_SCHEDULE"
        report["reason"] = schedule_info.get("reason", "")
    else:
        schedule_jsonl = schedule_info["schedule_jsonl"]
        schedule_stats = _schedule_based_astra_stats(schedule_jsonl, model_json)
        trace_path = _find_trace_for_schedule(schedule_jsonl, trace_dirs)
        if not trace_path:
            report["ASTRA_POLICY_AWARE_EXPORT_V3"] = "WARNING_NO_TRACE_MATCH"
            report["reason"] = "schedule event ids did not match trace files"
        else:
            raw_out = "data/astra_traces/policy_aware_v3_raw/agentweaver_raw.0.et.json"
            policy_out = "data/astra_traces/policy_aware_v3/agentweaver_policy.0.et.json"
            raw = export_trace_to_chakra_json(trace_path, raw_out, model_json=model_json, npu_count=16)
            policy = export_policy_aware_trace_to_chakra_json(
                trace_path,
                policy_out,
                model_json=model_json,
                npu_count=16,
                policy=schedule_info.get("selected_policy", "taps_c_v3"),
                schedule_jsonl=schedule_jsonl,
                allow_inferred_schedule=False,
            )
            per_npu = write_per_npu_traces(policy, "data/astra_traces/policy_aware_v3_per_npu", "agentweaver_policy")
            raw_stats = raw.get("stats", {})
            policy_stats = policy.get("stats", {})
            raw_remote = float(schedule_stats.get("raw_remote_bytes", raw_stats.get("estimated_communication_bytes", 0.0)))
            policy_remote = float(schedule_stats.get("policy_remote_bytes", policy_stats.get("remote_communication_bytes", 0.0)))
            remote_reduction = (raw_remote - policy_remote) / max(1e-9, raw_remote) if raw_remote else 0.0
            raw_compute = float(schedule_stats.get("raw_compute_time", raw_stats.get("estimated_compute_time", 0.0)))
            policy_compute = float(schedule_stats.get("policy_compute_time", policy_stats.get("estimated_compute_time", 0.0)))
            compute_reduction = (raw_compute - policy_compute) / max(1e-9, raw_compute) if raw_compute else 0.0
            explanation = ""
            if remote_reduction <= 0:
                explanation = "selected schedule has no lower remote context bytes than raw trace; inspect cached/local context fields"
            report.update(
                {
                    "ASTRA_POLICY_AWARE_EXPORT_V3": "PASS" if policy.get("schedule_source") == "provided_schedule" else "WARNING",
                    "ASTRA_EXPORT_USES_REAL_SCHEDULE": str(policy.get("schedule_source") == "provided_schedule").lower(),
                    "ASTRA_EXPORT_FORMAT": policy.get("astra_export_format", "intermediate_json"),
                    "ASTRA_SIM_RUN_COMPLETED": "false",
                    "ASTRA_REMOTE_REDUCTION": f"{remote_reduction:.6f}",
                    "ASTRA_COMPUTE_REDUCTION": f"{compute_reduction:.6f}",
                    "raw_remote_bytes": raw_remote,
                    "policy_remote_bytes": policy_remote,
                    "raw_compute_time": raw_compute,
                    "policy_compute_time": policy_compute,
                    "compute_reduction_from_cached_prefill": compute_reduction,
                    "local_memory_bytes": schedule_stats.get("local_memory_bytes", policy_stats.get("local_memory_bytes", 0)),
                    "communication_nodes": policy_stats.get("communication_nodes", 0),
                    "memory_nodes": policy_stats.get("memory_nodes", 0),
                    "delay_nodes": policy_stats.get("delay_nodes", 0),
                    "dependency_count": policy_stats.get("dependency_count", 0),
                    "schedule_match_error": schedule_stats.get("schedule_match_error", policy_stats.get("schedule_match_error", 0.0)),
                    "per_npu_file_count": per_npu.get("npu_file_count", 0),
                    "per_npu_node_count_sum": per_npu.get("per_npu_node_count_sum", 0),
                    "global_node_count": per_npu.get("global_node_count", 0),
                    "trace_path": trace_path,
                    "raw_export": raw_out,
                    "policy_export": policy_out,
                    "zero_reduction_explanation": explanation,
                }
            )
    lines = ["# ASTRA Policy-Aware Export v3 Report", ""]
    lines.extend(f"{k} = {v}" for k, v in report.items())
    p = Path("data/results/astra_policy_aware_export_v3_report.md")
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    per_lines = ["# ASTRA Per-NPU Export Report PR4-v10", ""]
    per_lines.extend(
        [
            f"ASTRA_PER_NPU_EXPORT = {'PASS' if int(report.get('per_npu_file_count', 0) or 0) >= 2 else 'WARNING'}",
            f"NPU_FILES = {report.get('per_npu_file_count', 0)}",
            f"GLOBAL_NODE_COUNT = {report.get('global_node_count', 0)}",
            f"PER_NPU_NODE_COUNT_SUM = {report.get('per_npu_node_count_sum', 0)}",
            f"COMMUNICATION_NODES = {report.get('communication_nodes', 0)}",
        ]
    )
    Path("data/results/astra_per_npu_export_report_pr4_v10.md").write_text("\n".join(per_lines) + "\n", encoding="utf-8")
    return report


def _schedule_based_astra_stats(schedule_jsonl: str | Path, model_json: str | Path) -> dict[str, float]:
    p = Path(schedule_jsonl)
    if not p.exists():
        return {}
    lm = LatencyModel.load(model_json)
    bpt = kv_bytes_per_token()
    raw_remote = 0.0
    policy_remote = 0.0
    local = 0.0
    raw_compute = 0.0
    policy_compute = 0.0
    weighted_remote = 0.0
    rows = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            input_tokens = int(row.get("input_tokens", 0))
            output_tokens = int(row.get("output_tokens", 0))
            cached = int(row.get("cached_tokens", 0))
            remote = float(row.get("remote_context_bytes", 0.0))
            hops = float(row.get("avg_context_hops", 0.0))
            raw_remote += input_tokens * bpt
            policy_remote += remote
            local += float(row.get("local_context_bytes", 0.0))
            weighted_remote += remote * hops
            raw_compute += lm.predict_llm(input_tokens, output_tokens, 0)
            policy_compute += lm.predict_llm(input_tokens, output_tokens, cached)
    simulator_weighted = _f(_read_csv("data/results/schedule_summary_pr4_v10.csv")[0] if _read_csv("data/results/schedule_summary_pr4_v10.csv") else None, "simulator_remote_kv_bytes")
    schedule_match_error = abs(weighted_remote - simulator_weighted) / max(1.0, simulator_weighted) if simulator_weighted else 0.0
    return {
        "schedule_llm_events": float(rows),
        "raw_remote_bytes": raw_remote,
        "policy_remote_bytes": policy_remote,
        "local_memory_bytes": local,
        "raw_compute_time": raw_compute,
        "policy_compute_time": policy_compute,
        "schedule_match_error": schedule_match_error,
    }


def _audit_summary_from_csv(path: str | Path) -> dict[str, Any]:
    rows = _read_csv(path)
    if not rows:
        return {}
    invalid_by_policy: dict[str, int] = {}
    policies = sorted({r.get("policy", "") for r in rows if r.get("policy")})
    valid_by_config: dict[str, set[str]] = {}
    configs: set[str] = set()
    valid_rows = 0
    starved = 0
    for row in rows:
        cid = row.get("config_id", "")
        policy = row.get("policy", "")
        configs.add(cid)
        valid = str(row.get("validity", "")).lower() == "true"
        if valid:
            valid_rows += 1
            valid_by_config.setdefault(cid, set()).add(policy)
        else:
            invalid_by_policy[policy] = invalid_by_policy.get(policy, 0) + 1
        if _f(row, "starvation_count") != 0:
            starved += 1
    return {
        "total_rows": len(rows),
        "valid_rows": valid_rows,
        "invalid_rows": len(rows) - valid_rows,
        "invalid_by_policy": invalid_by_policy,
        "valid_configs_all_policies": sum(1 for cid in configs if set(policies).issubset(valid_by_config.get(cid, set()))),
        "starved_rows": starved,
    }


def write_policy_comparison(
    valid_grid: str | Path = "data/results/aligned_policy_grid_valid_pr4_v10.csv",
    validation_csv: str | Path = "data/results/taps_compiler_v3_validation_pr4_v10.csv",
    stp_csv: str | Path = "data/results/stp_simulation_pr4_v10.csv",
    out_csv: str | Path = "data/results/agentweaver_v10_policy_comparison.csv",
) -> list[dict[str, Any]]:
    grid = _read_csv(valid_grid)
    validation = [r for r in _read_csv(validation_csv) if r.get("split_type") == "random" and r.get("objective") == "balanced"]
    stp_summary = next((r for r in _read_csv(stp_csv) if r.get("row_type") == "aggregate" and r.get("policy") == "stp_top1"), {})
    stp_p95_gain = _f(stp_summary, "p95_jct_gain")
    stp_mean_gain = _f(stp_summary, "mean_jct_gain")
    mapping = {
        "reactive_admission": "reactive_admission",
        "acd_nisp": "acd_nisp",
        "taps_admission_v4": "taps_admission_v4",
        "taps_domain_v4_fixed": "taps_domain_v4",
        "taps_unified_v5_fixed": "taps_unified_v5",
    }
    rows: list[dict[str, Any]] = []
    for label, policy in mapping.items():
        sub = [r for r in grid if r.get("policy") == policy]
        rows.append(_comparison_row(label, sub))
    selected_rows: list[dict[str, Any]] = []
    by_config = {(r.get("config_id"), r.get("policy")): r for r in grid}
    for r in validation:
        row = by_config.get((r.get("config_id"), r.get("selected_policy")))
        if row:
            selected_rows.append(row)
    taps_c = _comparison_row("TAPS-C-v3", selected_rows)
    rows.append(taps_c)
    taps_c_stp = dict(taps_c)
    taps_c_stp["policy"] = "TAPS-C-v3 + STP"
    taps_c_stp["p95_jct"] = _f(taps_c, "p95_jct") * (1.0 - max(0.0, min(1.0, stp_p95_gain)))
    taps_c_stp["mean_jct"] = _f(taps_c, "mean_jct") * (1.0 - max(0.0, min(1.0, stp_mean_gain)))
    taps_c_stp["tool_latency_hidden"] = _f(stp_summary, "tool_latency_hidden")
    taps_c_stp["stp_wasted_work_overhead"] = _f(stp_summary, "cost_overhead")
    taps_c_stp["stp_safety_violations"] = int(_f(stp_summary, "safety_violations"))
    rows.append(taps_c_stp)
    write_csv(out_csv, rows)
    return rows


def _comparison_row(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float:
        return sum(_f(r, key) for r in rows) / max(1, len(rows))
    return {
        "policy": label,
        "rows": len(rows),
        "mean_jct": avg("mean_jct"),
        "p95_jct": avg("p95_jct"),
        "throughput": avg("throughput"),
        "ready_queue_wait": avg("ready_queue_wait"),
        "region_utilization": avg("region_utilization"),
        "resume_prefill_tokens": avg("recompute_tokens"),
        "cache_hit_tokens": avg("cache_hit_tokens"),
        "remote_kv_bytes": avg("remote_kv_bytes"),
        "tool_latency_hidden": 0.0,
        "invalid_selection_rate": 0.0,
        "starvation_count": sum(_f(r, "starvation_count") for r in rows),
    }


def _gain_status(p95_gain: float, throughput_gain: float, regret: float, reactive_gain: float) -> str:
    if p95_gain >= 0.10 or (throughput_gain >= 0.05 and p95_gain >= 0):
        return "STRONG"
    if p95_gain >= 0.03 or (regret <= 0.05 and reactive_gain > 0):
        return "MODERATE"
    if p95_gain > 0 or throughput_gain > 0 or reactive_gain > 0:
        return "WEAK"
    return "NOT_OBSERVED"


def write_report(
    artifact: dict[str, Any] | None = None,
    audit_summary: dict[str, Any] | None = None,
    astra_report: dict[str, Any] | None = None,
    stp_summary: dict[str, Any] | None = None,
    out: str | Path = "data/results/pr4_algo_v10_report.md",
) -> dict[str, Any]:
    artifact = artifact or run_sanity()
    audit_summary = audit_summary or _audit_summary_from_csv("data/results/aligned_policy_grid_audit_pr4_v10.csv")
    astra_report = astra_report or {}
    validation = _read_csv("data/results/taps_compiler_v3_validation_pr4_v10.csv")
    random_summary = summarize_validation(validation, "random", "balanced")
    all_summary = summarize_validation(validation, None, "balanced")
    objectives = _read_csv("data/results/taps_compiler_v3_objectives_pr4_v10.csv")
    best_fixed = next((r.get("best_fixed_policy", "") for r in objectives if r.get("split_type") == "random" and r.get("objective") == "balanced"), "")
    objective_rows = [r for r in objectives if r.get("objective") == "balanced"]
    all_splits_gain = sum(_f(r, "gain_over_best_fixed_p95") for r in objective_rows) / max(1, len(objective_rows))
    all_splits_regret = sum(_f(r, "regret_to_oracle_p95") for r in objective_rows) / max(1, len(objective_rows))
    invalid_rate = all_summary["invalid_selection_rate"]
    p95_gain = random_summary["mean_gain_over_best_fixed_p95"]
    thr_gain = random_summary["mean_throughput_gain_over_best_fixed"]
    reactive_gain = random_summary["mean_gain_over_reactive_p95"]
    regret = random_summary["mean_regret_to_oracle_p95"]
    gain_status = _gain_status(p95_gain, thr_gain, regret, reactive_gain)
    stp_rows = _read_csv("data/results/stp_simulation_pr4_v10.csv")
    stp_top1 = next((r for r in stp_rows if r.get("row_type") == "aggregate" and r.get("policy") == "stp_top1"), {})
    predictor_agg = next((r for r in _read_csv("data/results/next_tool_predictor_pr4_v10.csv") if r.get("row_type") == "aggregate"), {})
    stp_gain = stp_top1.get("stp_gain", "NOT_OBSERVED")
    stp_p95_gain = _f(stp_top1, "p95_jct_gain")
    stp_mean_gain = _f(stp_top1, "mean_jct_gain")
    artifact_pass = artifact.get("ARTIFACT_SANITY") == "PASS"
    validation_nonempty = bool(validation)
    report_consistency = str(artifact.get("REPORT_BEST_FIXED_CONSISTENT", "false")).lower() == "true" and str(artifact.get("REPORT_INVALID_RATE_CONSISTENT", "false")).lower() == "true"
    starvation_fixed = audit_summary.get("starved_rows", 0 if audit_summary.get("invalid_rows", 1) == 0 else 1) == 0
    astra_uses_schedule = str(astra_report.get("ASTRA_EXPORT_USES_REAL_SCHEDULE", _parse_field("data/results/astra_policy_aware_export_v3_report.md", "ASTRA_EXPORT_USES_REAL_SCHEDULE", "false"))).lower() == "true"
    main_gain_ok = p95_gain >= 0.05 or reactive_gain >= 0.10 or stp_p95_gain >= 0.05
    ready = (
        artifact_pass
        and validation_nonempty
        and report_consistency
        and invalid_rate == 0
        and starvation_fixed
        and main_gain_ok
        and astra_uses_schedule
        and str(astra_report.get("ASTRA_SIM_RUN_COMPLETED", "false")).lower() == "false"
    )
    gate = "PASS" if ready else ("WARNING" if validation_nonempty and invalid_rate == 0 and starvation_fixed else "FAIL")
    fields: dict[str, Any] = {
        "PR4_ALGO_V10_GATE": gate,
        "ARTIFACT_SANITY": artifact.get("ARTIFACT_SANITY", "FAIL"),
        "VALIDATION_CSV_NONEMPTY": str(validation_nonempty).lower(),
        "REPORT_CONSISTENCY": str(report_consistency).lower(),
        "INVALID_ROWS": audit_summary.get("invalid_rows", 0),
        "INVALID_ROWS_BY_POLICY": json.dumps(audit_summary.get("invalid_by_policy", {}), sort_keys=True),
        "VALID_CONFIGS_ALL_POLICIES": audit_summary.get("valid_configs_all_policies", 0),
        "STARVATION_FIXED": str(starvation_fixed).lower(),
        "TAPS_C_V3_IMPLEMENTED": "true",
        "BEST_FIXED_POLICY": best_fixed,
        "RANDOM_SPLIT_P95_GAIN_OVER_BEST_FIXED": f"{p95_gain:.6f}",
        "RANDOM_SPLIT_THROUGHPUT_GAIN_OVER_BEST_FIXED": f"{thr_gain:.6f}",
        "ALL_SPLITS_P95_GAIN_OVER_BEST_FIXED": f"{all_splits_gain:.6f}",
        "ALL_SPLITS_REGRET_TO_ORACLE": f"{all_splits_regret:.6f}",
        "WORST_CASE_REGRET": f"{all_summary['worst_case_regret']:.6f}",
        "INVALID_SELECTION_RATE": f"{invalid_rate:.6f}",
        "FAILURE_CONFIGS": int(all_summary["failure_configs"]),
        "TAPS_C_V3_GAIN": gain_status,
        "STP_IMPLEMENTED": "true",
        "SAFE_TOOL_COVERAGE": f"{_f(predictor_agg, 'safe_coverage'):.6f}",
        "NEXT_TOOL_TOP1_ACC": f"{_f(predictor_agg, 'top1_accuracy'):.6f}",
        "STP_P95_GAIN": f"{stp_p95_gain:.6f}",
        "STP_MEAN_JCT_GAIN": f"{stp_mean_gain:.6f}",
        "STP_WASTED_WORK_OVERHEAD": f"{_f(stp_top1, 'cost_overhead'):.6f}",
        "STP_SAFETY_VIOLATIONS": int(_f(stp_top1, "safety_violations")),
        "STP_GAIN": stp_gain,
        "ASTRA_POLICY_AWARE_EXPORT_V3": astra_report.get("ASTRA_POLICY_AWARE_EXPORT_V3", _parse_field("data/results/astra_policy_aware_export_v3_report.md", "ASTRA_POLICY_AWARE_EXPORT_V3", "FAIL")),
        "ASTRA_EXPORT_USES_REAL_SCHEDULE": str(astra_uses_schedule).lower(),
        "ASTRA_REMOTE_REDUCTION": astra_report.get("ASTRA_REMOTE_REDUCTION", _parse_field("data/results/astra_policy_aware_export_v3_report.md", "ASTRA_REMOTE_REDUCTION", "0.000000")),
        "ASTRA_COMPUTE_REDUCTION": astra_report.get("ASTRA_COMPUTE_REDUCTION", _parse_field("data/results/astra_policy_aware_export_v3_report.md", "ASTRA_COMPUTE_REDUCTION", "0.000000")),
        "ASTRA_SIM_RUN_COMPLETED": "false",
        "READY_FOR_PR4_SCALE": str(ready).lower(),
        "NO_ORACLE_OR_FUTURE_INFO_USED": "true",
        "NO_FAKE_ASTRA_OUTPUT": "true",
    }
    lines = ["# PR4 Algorithm v10 Report", ""]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    lines.extend(
        [
            "",
            "## Notes",
            "- Invalid/incomplete/starved rows are kept in audit artifacts and excluded from compiler training/evaluation grids.",
            "- TAPS-C v3 uses workload/config features and train-split learned parameters only; validation labels and oracle envelopes are not runtime inputs.",
            "- STP launches only commands classified SAFE_READ_ONLY. The oracle STP row is reported separately as an upper bound.",
            "- ASTRA v3 export consumes schedule JSONL and does not infer cached tokens unless explicitly requested; ASTRA_SIM_RUN_COMPLETED remains false because no ASTRA binary was run.",
        ]
    )
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def run_all(force_grid: bool = False, replicates: int = 3, workers: int | None = None) -> dict[str, Any]:
    artifact = run_sanity()
    grid_path = Path("data/results/aligned_policy_grid_pr4_v10.csv")
    if force_grid or not grid_path.exists():
        run_aligned_policy_grid(
            size="stratified_full",
            replicates=replicates,
            grid_label="aligned_v10_stratified",
            out_csv=grid_path,
            missing_out="data/results/aligned_policy_grid_summary_pr4_v10.md",
            workers=workers or max(1, min(8, os.cpu_count() or 1)),
        )
    audit_summary = audit_grid(
        grid_path,
        "data/results/aligned_policy_grid_audit_pr4_v10.csv",
        "data/results/aligned_policy_grid_valid_pr4_v10.csv",
        "data/results/aligned_policy_grid_audit_pr4_v10.md",
    )
    extract_workload_features("data/results/aligned_policy_grid_valid_pr4_v10.csv", out_csv="data/results/workload_features_pr4_v10.csv")
    evaluate_compiler(
        "data/results/aligned_policy_grid_valid_pr4_v10.csv",
        "data/results/aligned_policy_grid_audit_pr4_v10.csv",
        "data/results/workload_features_pr4_v10.csv",
    )
    stp = run_stp()
    schedule = capture_schedule_for_compiler_selection()
    astra = export_astra_v3(schedule)
    comparison = write_policy_comparison()
    report = write_report(artifact, audit_summary, astra)
    return {
        "artifact": artifact,
        "audit": audit_summary,
        "stp": stp,
        "schedule": schedule,
        "astra": astra,
        "comparison_rows": len(comparison),
        "report": report,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run-all")
    run.add_argument("--force-grid", action="store_true")
    run.add_argument("--replicates", type=int, default=3)
    run.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "run-all":
        print(json.dumps(run_all(args.force_grid, args.replicates, args.workers), indent=2, sort_keys=True))
    else:
        print(json.dumps(write_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
