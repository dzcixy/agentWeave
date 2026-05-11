from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from agentweaver.analysis.policy_grid_audit import audit_grid
from agentweaver.analysis.pr4_v9_eval_methodology import write_methodology
from agentweaver.analysis.speculative_tool_prefetch_potential import analyze_prefetch_potential
from agentweaver.astra.export_chakra import (
    export_policy_aware_trace_to_chakra_json,
    export_trace_to_chakra_json,
    write_per_npu_traces,
)
from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.aligned_policy_sweep import ALIGNED_POLICIES, run_aligned_policy_grid
from agentweaver.simulator.taps_cost_model_v2 import evaluate_compiler, summarize_validation
from agentweaver.simulator.taps_unified import TAPSUnifiedReplay, _load_traces
from agentweaver.simulator.workload_feature_extractor import extract_workload_features
from agentweaver.tracing.trace_schema import Trace
from agentweaver.utils.io import ensure_dir


SUPPORTED_SCHEDULE_POLICIES = {
    "reactive_admission": "reactive_admission",
    "acd_nisp": "acd_nisp",
    "taps_domain_v4": "taps_domain_v4",
    "taps_admission_v4": "taps_admission_v4",
    "taps_unified_v5": "taps_unified",
}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
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


def _config_row(rows: list[dict[str, str]], cid: str, policy: str) -> dict[str, str] | None:
    return next((r for r in rows if r.get("config_id") == cid and r.get("policy") == policy), None)


def _find_trace_for_schedule(schedule_jsonl: str | Path, trace_dirs: list[str | Path]) -> str | None:
    event_ids: set[str] = set()
    with Path(schedule_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                event_ids.add(str(json.loads(line).get("event_id", "")))
    for trace_dir in trace_dirs:
        for path in sorted(Path(trace_dir).glob("*.jsonl")):
            trace = Trace.from_jsonl(path)
            if any(ev.event_id in event_ids for ev in trace.events):
                return str(path)
    return None


def capture_schedule_for_compiler_selection(
    validation_csv: str | Path,
    valid_grid_csv: str | Path,
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    run_id: str = "pr4_v9_taps_c",
) -> dict[str, Any]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    validation = _read_csv(validation_csv)
    grid = _read_csv(valid_grid_csv)
    chosen: dict[str, str] | None = None
    for row in validation:
        policy = row.get("selected_policy", "")
        if row.get("objective") == "balanced" and row.get("split_type") == "random" and policy in SUPPORTED_SCHEDULE_POLICIES:
            chosen = row
            break
    if not chosen:
        for row in validation:
            if row.get("selected_policy", "") in SUPPORTED_SCHEDULE_POLICIES:
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
    replay = TAPSUnifiedReplay(
        traces,
        int(_f(cfg, "total_sessions")),
        int(_f(cfg, "active_session_limit")),
        int(_f(cfg, "effective_regions")),
        str(cfg.get("arrival_pattern", "closed_loop")),
        int(_f(cfg, "memory_budget_gb")),
        SUPPORTED_SCHEDULE_POLICIES[selected_policy],
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
        "replay_policy": SUPPORTED_SCHEDULE_POLICIES[selected_policy],
        "completed_sessions": result.get("completed_sessions", 0),
        "starvation_count": result.get("starvation_count", 0),
    }


def export_astra_v2(
    schedule_info: dict[str, Any],
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
) -> dict[str, Any]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    report = {
        "ASTRA_POLICY_AWARE_EXPORT_V2": "FAIL",
        "ASTRA_EXPORT_USES_REAL_SCHEDULE": "false",
        "ASTRA_SIM_RUN_COMPLETED": "false",
        "ASTRA_REMOTE_REDUCTION": "0.000000",
    }
    if not schedule_info.get("schedule_available"):
        report["ASTRA_POLICY_AWARE_EXPORT_V2"] = "WARNING_NO_SCHEDULE"
        report["reason"] = schedule_info.get("reason", "")
    else:
        schedule_jsonl = schedule_info["schedule_jsonl"]
        trace_path = _find_trace_for_schedule(schedule_jsonl, trace_dirs)
        if not trace_path:
            report["ASTRA_POLICY_AWARE_EXPORT_V2"] = "WARNING_NO_TRACE_MATCH"
            report["reason"] = "schedule event ids did not match trace files"
        else:
            raw_out = "data/astra_traces/policy_aware_v2_raw/agentweaver_raw.0.et.json"
            policy_out = "data/astra_traces/policy_aware_v2/agentweaver_policy.0.et.json"
            raw = export_trace_to_chakra_json(trace_path, raw_out, model_json=model_json, npu_count=16)
            policy = export_policy_aware_trace_to_chakra_json(
                trace_path,
                policy_out,
                model_json=model_json,
                npu_count=16,
                policy=schedule_info.get("selected_policy", "taps_c_v2"),
                schedule_jsonl=schedule_jsonl,
                allow_inferred_schedule=False,
            )
            per_npu = write_per_npu_traces(policy, "data/astra_traces/policy_aware_v2_per_npu", "agentweaver_policy")
            raw_remote = float(raw.get("stats", {}).get("estimated_communication_bytes", 0.0))
            policy_remote = float(policy.get("stats", {}).get("remote_communication_bytes", 0.0))
            reduction = (raw_remote - policy_remote) / max(1e-9, raw_remote) if raw_remote else 0.0
            report.update(
                {
                    "ASTRA_POLICY_AWARE_EXPORT_V2": "PASS" if policy.get("schedule_source") == "provided_schedule" else "WARNING",
                    "ASTRA_EXPORT_USES_REAL_SCHEDULE": str(policy.get("schedule_source") == "provided_schedule").lower(),
                    "ASTRA_EXPORT_FORMAT": policy.get("astra_export_format", "intermediate_json"),
                    "ASTRA_SIM_RUN_COMPLETED": "false",
                    "ASTRA_REMOTE_REDUCTION": f"{reduction:.6f}",
                    "raw_remote_bytes": raw_remote,
                    "policy_remote_bytes": policy_remote,
                    "local_memory_bytes": policy.get("stats", {}).get("local_memory_bytes", 0),
                    "communication_nodes": policy.get("stats", {}).get("communication_nodes", 0),
                    "memory_nodes": policy.get("stats", {}).get("memory_nodes", 0),
                    "delay_nodes": policy.get("stats", {}).get("delay_nodes", 0),
                    "dependency_count": policy.get("stats", {}).get("dependency_count", 0),
                    "per_npu_file_count": per_npu.get("npu_file_count", 0),
                    "per_npu_node_count_sum": per_npu.get("per_npu_node_count_sum", 0),
                    "global_node_count": per_npu.get("global_node_count", 0),
                    "cross_npu_communication_nodes": per_npu.get("cross_npu_communication_nodes", 0),
                    "trace_path": trace_path,
                    "raw_export": raw_out,
                    "policy_export": policy_out,
                }
            )
    lines = ["# ASTRA Policy-Aware Export v2 Report", ""]
    lines.extend(f"{k} = {v}" for k, v in report.items())
    p = Path("data/results/astra_policy_aware_export_v2_report.md")
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    per_lines = ["# ASTRA Per-NPU Export Report PR4-v9", ""]
    per_lines.extend(
        [
            f"ASTRA_PER_NPU_EXPORT = {'PASS' if int(report.get('per_npu_file_count', 0) or 0) >= 2 else 'WARNING'}",
            f"NPU_FILES = {report.get('per_npu_file_count', 0)}",
            f"GLOBAL_NODE_COUNT = {report.get('global_node_count', 0)}",
            f"PER_NPU_NODE_COUNT_SUM = {report.get('per_npu_node_count_sum', 0)}",
            f"CROSS_NPU_COMMUNICATION_NODES = {report.get('cross_npu_communication_nodes', 0)}",
        ]
    )
    Path("data/results/astra_per_npu_export_report_pr4_v9.md").write_text("\n".join(per_lines) + "\n", encoding="utf-8")
    return report


def _dict_from_md_counts(path: str | Path, heading: str) -> str:
    p = Path(path)
    if not p.exists():
        return "{}"
    lines = p.read_text(encoding="utf-8").splitlines()
    capture = False
    out: dict[str, int] = {}
    for line in lines:
        if line.strip() == heading:
            capture = True
            continue
        if capture and line.startswith("## "):
            break
        if capture and line.startswith("- ") and ":" in line:
            k, v = line[2:].split(":", 1)
            try:
                out[k.strip()] = int(v.strip())
            except Exception:
                pass
    return json.dumps(out, sort_keys=True)


def _audit_summary_from_csv(path: str | Path) -> dict[str, Any]:
    rows = _read_csv(path)
    if not rows:
        return {}
    invalid_by_policy: dict[str, int] = {}
    policies = sorted({r.get("policy", "") for r in rows if r.get("policy")})
    valid_by_config: dict[str, set[str]] = {}
    configs: set[str] = set()
    valid_rows = 0
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
    return {
        "total_rows": len(rows),
        "valid_rows": valid_rows,
        "invalid_rows": len(rows) - valid_rows,
        "invalid_by_policy": invalid_by_policy,
        "valid_configs_all_policies": sum(1 for cid in configs if set(policies).issubset(valid_by_config.get(cid, set()))),
    }


def _gain_status(p95_gain: float, throughput_gain: float, regret: float, reactive_gain: float) -> str:
    if p95_gain >= 0.10 or (throughput_gain >= 0.05 and p95_gain >= 0):
        return "STRONG"
    if p95_gain >= 0.03 or (throughput_gain >= 0.03 and p95_gain >= 0) or (regret <= 0.05 and reactive_gain > 0):
        return "MODERATE"
    if p95_gain > 0 or throughput_gain > 0 or reactive_gain > 0:
        return "WEAK"
    return "NOT_OBSERVED"


def write_report(
    audit_summary: dict[str, Any] | None = None,
    stratified_summary: dict[str, Any] | None = None,
    astra_report: dict[str, Any] | None = None,
    out: str | Path = "data/results/pr4_algo_v9_report.md",
) -> dict[str, Any]:
    audit_summary = audit_summary or _audit_summary_from_csv("data/results/aligned_policy_grid_audit_pr4_v9.csv")
    stratified_summary = stratified_summary or _audit_summary_from_csv("data/results/aligned_policy_grid_stratified_audit_pr4_v9.csv") or audit_summary
    astra_report = astra_report or {}
    validation = _read_csv("data/results/taps_compiler_v2_validation_pr4_v9.csv")
    random_summary = summarize_validation(validation, "random")
    all_summary = summarize_validation(validation)
    objectives = _read_csv("data/results/taps_compiler_v2_objectives_pr4_v9.csv")
    best_fixed = next((r.get("best_fixed_policy", "") for r in objectives if r.get("split_type") == "random" and r.get("objective") == "balanced"), "")
    p95_gain = random_summary["mean_gain_over_best_fixed_p95"]
    thr_gain = random_summary["mean_throughput_gain_over_best_fixed"]
    regret = random_summary["mean_regret_to_oracle_p95"]
    reactive_gain = random_summary["mean_gain_over_reactive_p95"]
    invalid_rate = all_summary["invalid_selection_rate"]
    gain_status = _gain_status(p95_gain, thr_gain, regret, reactive_gain)
    astra_status = astra_report.get("ASTRA_POLICY_AWARE_EXPORT_V2", _parse_field("data/results/astra_policy_aware_export_v2_report.md", "ASTRA_POLICY_AWARE_EXPORT_V2", "FAIL"))
    astra_uses_schedule = str(astra_report.get("ASTRA_EXPORT_USES_REAL_SCHEDULE", _parse_field("data/results/astra_policy_aware_export_v2_report.md", "ASTRA_EXPORT_USES_REAL_SCHEDULE", "false"))).lower() == "true"
    ready = (
        invalid_rate == 0
        and (p95_gain >= 0.03 or (regret <= 0.05 and reactive_gain > 0))
        and astra_uses_schedule
        and astra_status in {"PASS", "WARNING"}
    )
    gate = "PASS" if ready else ("WARNING" if validation else "FAIL")
    spec_status = _parse_field("data/results/speculative_tool_prefetch_potential_pr4_v9.md", "SPECULATIVE_TOOL_PREFETCH_ANALYSIS", "NOT_RUN")
    fields: dict[str, Any] = {
        "PR4_ALGO_V9_GATE": gate,
        "VALID_ROWS": stratified_summary.get("valid_rows", audit_summary.get("valid_rows", 0)),
        "INVALID_ROWS": stratified_summary.get("invalid_rows", audit_summary.get("invalid_rows", 0)),
        "INVALID_ROWS_BY_POLICY": json.dumps(stratified_summary.get("invalid_by_policy", audit_summary.get("invalid_by_policy", {})), sort_keys=True),
        "VALID_CONFIGS_ALL_POLICIES": stratified_summary.get("valid_configs_all_policies", audit_summary.get("valid_configs_all_policies", 0)),
        "STRATIFIED_GRID_CONFIGS": len({r.get("config_id") for r in _read_csv("data/results/aligned_policy_grid_stratified_pr4_v9.csv")}),
        "TAPS_C_V2_IMPLEMENTED": "true",
        "BEST_FIXED_POLICY": best_fixed,
        "RANDOM_SPLIT_P95_GAIN_OVER_BEST_FIXED": f"{p95_gain:.6f}",
        "RANDOM_SPLIT_THROUGHPUT_GAIN_OVER_BEST_FIXED": f"{thr_gain:.6f}",
        "RANDOM_SPLIT_REGRET_TO_ORACLE": f"{regret:.6f}",
        "ALL_SPLITS_P95_GAIN_OVER_BEST_FIXED": f"{all_summary['mean_gain_over_best_fixed_p95']:.6f}",
        "ALL_SPLITS_REGRET_TO_ORACLE": f"{all_summary['mean_regret_to_oracle_p95']:.6f}",
        "INVALID_SELECTION_RATE": f"{invalid_rate:.6f}",
        "WORST_CASE_REGRET": f"{all_summary['worst_case_regret']:.6f}",
        "FAILURE_CONFIGS": int(all_summary["failure_configs"]),
        "TAPS_C_V2_GAIN": gain_status,
        "ASTRA_POLICY_AWARE_EXPORT_V2": astra_status,
        "ASTRA_EXPORT_USES_REAL_SCHEDULE": str(astra_uses_schedule).lower(),
        "ASTRA_REMOTE_REDUCTION": astra_report.get("ASTRA_REMOTE_REDUCTION", _parse_field("data/results/astra_policy_aware_export_v2_report.md", "ASTRA_REMOTE_REDUCTION", "0.000000")),
        "ASTRA_SIM_RUN_COMPLETED": "false",
        "SPECULATIVE_TOOL_PREFETCH_ANALYSIS": spec_status,
        "SAFE_TOOL_COVERAGE": _parse_field("data/results/speculative_tool_prefetch_potential_pr4_v9.md", "SAFE_TOOL_COVERAGE", "0.000000"),
        "NEXT_TOOL_TOP1_ACC": _parse_field("data/results/speculative_tool_prefetch_potential_pr4_v9.md", "NEXT_TOOL_TOP1_ACC", "0.000000"),
        "POTENTIAL_LATENCY_SAVED": _parse_field("data/results/speculative_tool_prefetch_potential_pr4_v9.md", "POTENTIAL_LATENCY_SAVED", "0.000000"),
        "READY_FOR_PR4_SCALE": str(ready).lower(),
    }
    lines = ["# PR4 Algorithm v9 Report", ""]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    lines.extend(
        [
            "",
            "## Notes",
            "- Invalid/incomplete/starved policy rows are audited and excluded from best-fixed, oracle-envelope, and default TAPS-C v2 training.",
            "- Oracle envelope is only an upper bound; it is not a deployable baseline.",
            "- TAPS-C v2 uses workload-only features and train-split pairwise/dominance models. Validation labels are not used at runtime.",
            "- ASTRA policy-aware export consumes the simulator schedule JSONL. It does not infer cached tokens from policy names.",
        ]
    )
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def run_all(stratified: bool = True, force_grid: bool = False, replicates: int = 1) -> dict[str, Any]:
    v8_audit = audit_grid()
    strat_audit = {}
    valid_grid_for_compiler = "data/results/aligned_policy_grid_valid_pr4_v9.csv"
    if stratified:
        strat_path = Path("data/results/aligned_policy_grid_stratified_pr4_v9.csv")
        if force_grid or not strat_path.exists():
            run_aligned_policy_grid(
                size="stratified_full",
                replicates=replicates,
                grid_label="aligned_v9_stratified",
                out_csv=strat_path,
                missing_out="data/results/aligned_policy_grid_stratified_summary_pr4_v9.md",
            )
        strat_audit = audit_grid(
            strat_path,
            "data/results/aligned_policy_grid_stratified_audit_pr4_v9.csv",
            "data/results/aligned_policy_grid_stratified_valid_pr4_v9.csv",
            "data/results/aligned_policy_grid_stratified_audit_pr4_v9.md",
        )
        valid_grid_for_compiler = "data/results/aligned_policy_grid_stratified_valid_pr4_v9.csv"
    write_methodology(valid_grid_for_compiler)
    extract_workload_features(valid_grid_for_compiler)
    evaluate_compiler(valid_grid_for_compiler, "data/results/aligned_policy_grid_stratified_audit_pr4_v9.csv" if stratified else "data/results/aligned_policy_grid_audit_pr4_v9.csv")
    schedule = capture_schedule_for_compiler_selection("data/results/taps_compiler_v2_validation_pr4_v9.csv", valid_grid_for_compiler)
    astra = export_astra_v2(schedule)
    analyze_prefetch_potential()
    report = write_report(v8_audit, strat_audit or v8_audit, astra)
    return {"v8_audit": v8_audit, "stratified_audit": strat_audit, "schedule": schedule, "astra": astra, "report": report}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run-all")
    run.add_argument("--no-stratified", action="store_true")
    run.add_argument("--force-grid", action="store_true")
    run.add_argument("--replicates", type=int, default=1)
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "run-all":
        print(json.dumps(run_all(not args.no_stratified, args.force_grid, args.replicates), indent=2, sort_keys=True))
    else:
        print(json.dumps(write_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
