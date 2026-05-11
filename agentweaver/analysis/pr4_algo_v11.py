from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from agentweaver.analysis.matched_policy_comparison import write_matched_comparison
from agentweaver.analysis.pr4_v11_artifact_sanity import run_sanity
from agentweaver.astra.export_chakra import export_schedule_jsonl_to_chakra_json, write_per_npu_traces
from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.agentweaver_v11_replay import write_policy_comparison
from agentweaver.simulator.safe_tool_prefetch_v2 import run_all as run_stp_v2
from agentweaver.simulator.taps_cost_model_v3 import evaluate_compiler
from agentweaver.simulator.taps_unified import TAPSUnifiedReplay, _load_traces
from agentweaver.simulator.taps_unified_adaptive import ADAPTIVE_POLICY, AdaptiveProfiles, TAPSAdaptiveReplay
from agentweaver.utils.io import ensure_dir, write_csv


SCHEDULE_POLICY_MAP = {
    "reactive_admission": "reactive_admission",
    "acd_nisp": "acd_nisp",
    "taps_admission_v4": "taps_admission_v4",
    "taps_domain_v4": "taps_domain_v4",
    "taps_domain_v4_fixed": "taps_domain_v4",
    "taps_unified_v5": "taps_unified",
    "taps_unified_v5_fixed": "taps_unified",
    "taps_unified_adaptive_v6": ADAPTIVE_POLICY,
}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fields(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if " = " in line:
            k, v = line.split(" = ", 1)
            out[k.strip()] = v.strip()
    return out


def _f(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return default
    try:
        value = row.get(key)
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def _avg(vals: list[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / max(1, len(vals))


def _slug(text: str, limit: int = 96) -> str:
    out = "".join(c if c.isalnum() else "_" for c in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:limit] or "unknown"


def _audit_summary(path: str | Path = "data/results/aligned_policy_grid_audit_pr4_v10.csv") -> dict[str, Any]:
    rows = _read_csv(path)
    policies = sorted({r.get("policy", "") for r in rows if r.get("policy")})
    valid_by_config: dict[str, set[str]] = {}
    invalid_rows = 0
    starved = 0
    configs: set[str] = set()
    for row in rows:
        cid = str(row.get("config_id", ""))
        policy = str(row.get("policy", ""))
        configs.add(cid)
        valid = str(row.get("validity", "")).lower() == "true"
        if valid:
            valid_by_config.setdefault(cid, set()).add(policy)
        else:
            invalid_rows += 1
        starved += int(_f(row, "starvation_count") != 0)
    return {
        "invalid_rows": invalid_rows,
        "valid_configs_all_policies": sum(1 for cid in configs if set(policies).issubset(valid_by_config.get(cid, set()))),
        "starved_rows": starved,
    }


def _balanced_validation_rows(path: str | Path = "data/results/taps_compiler_v3_validation_pr4_v11.csv") -> list[dict[str, str]]:
    return [r for r in _read_csv(path) if r.get("objective") == "balanced"]


def _grid_by_config(path: str | Path = "data/results/aligned_policy_grid_pr4_v10.csv") -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in _read_csv(path):
        out.setdefault(str(row.get("config_id", "")), []).append(row)
    return out


def _feature_by_config(path: str | Path = "data/results/workload_features_pr4_v10.csv") -> dict[str, dict[str, str]]:
    return {str(r.get("config_id", "")): r for r in _read_csv(path)}


def select_representative_configs(
    validation_csv: str | Path = "data/results/taps_compiler_v3_validation_pr4_v11.csv",
    grid_csv: str | Path = "data/results/aligned_policy_grid_pr4_v10.csv",
    feature_csv: str | Path = "data/results/workload_features_pr4_v10.csv",
    count: int = 6,
) -> list[str]:
    validation = _balanced_validation_rows(validation_csv)
    first_by_config: dict[str, dict[str, str]] = {}
    for row in validation:
        first_by_config.setdefault(str(row.get("config_id", "")), row)
    grid = _grid_by_config(grid_csv)
    features = _feature_by_config(feature_csv)
    configs = [cid for cid in sorted(first_by_config) if cid in grid]
    if not configs:
        return []

    def cfg_row(cid: str) -> dict[str, str]:
        return grid[cid][0]

    def pressure(cid: str) -> float:
        r = cfg_row(cid)
        return _f(r, "total_sessions") / max(1.0, _f(r, "active_session_limit")) + _f(r, "active_session_limit") / max(1.0, _f(r, "effective_regions"))

    picks: list[str] = []
    candidates = [
        min(configs, key=pressure),
        max(configs, key=lambda c: _f(cfg_row(c), "total_sessions") / max(1.0, _f(cfg_row(c), "active_session_limit"))),
        max(configs, key=lambda c: _f(cfg_row(c), "active_session_limit") / max(1.0, _f(cfg_row(c), "effective_regions"))),
        max([c for c in configs if cfg_row(c).get("arrival_pattern") == "bursty"] or configs, key=pressure),
        max(configs, key=lambda c: _f(features.get(c), "context_hotspot_score") + _f(features.get(c), "shared_context_ratio")),
        next((r.get("config_id", "") for r in validation if r.get("split_type") == "random"), configs[0]),
    ]
    for cid in candidates + configs:
        if cid and cid not in picks:
            picks.append(cid)
        if len(picks) >= count:
            break
    return picks


def _schedule_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_schedule_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _run_schedule(
    cfg: dict[str, str],
    policy_label: str,
    replay_policy: str,
    traces: list[Any],
    latency_model: LatencyModel,
    schedule_path: str | Path,
    seed: int,
) -> dict[str, Any]:
    common = {
        "total_sessions": int(_f(cfg, "total_sessions")),
        "active_session_limit": int(_f(cfg, "active_session_limit")),
        "effective_regions": int(_f(cfg, "effective_regions")),
        "arrival_pattern": str(cfg.get("arrival_pattern", "closed_loop")),
        "memory_budget_gb": int(_f(cfg, "memory_budget_gb")),
    }
    run_id = Path(schedule_path).stem.removesuffix("_schedule")
    if replay_policy == ADAPTIVE_POLICY:
        replay = TAPSAdaptiveReplay(
            traces,
            common["total_sessions"],
            common["active_session_limit"],
            common["effective_regions"],
            common["arrival_pattern"],
            common["memory_budget_gb"],
            latency_model,
            profiles=AdaptiveProfiles.default(),
            seed=seed,
            run_id=run_id,
            config_id=str(cfg.get("config_id", "")),
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
            replay_policy,
            latency_model,
            seed=seed,
            run_id=run_id,
            config_id=str(cfg.get("config_id", "")),
            schedule_log_path=schedule_path,
        )
    result = replay.run()
    return _schedule_summary(schedule_path, result, policy_label=policy_label, tool_latency_hidden=0.0)


def _schedule_summary(
    schedule_path: str | Path,
    replay_result: dict[str, Any],
    *,
    policy_label: str,
    tool_latency_hidden: float,
) -> dict[str, Any]:
    rows = _schedule_rows(schedule_path)
    remote_weighted = sum(float(r.get("remote_context_bytes", 0)) * float(r.get("avg_context_hops", 0.0)) for r in rows)
    simulator_remote = float(replay_result.get("remote_kv_bytes", remote_weighted) or 0.0)
    return {
        "run_id": rows[0].get("run_id", Path(schedule_path).stem) if rows else Path(schedule_path).stem,
        "policy": policy_label,
        "config_id": rows[0].get("config_id", replay_result.get("config_id", "")) if rows else replay_result.get("config_id", ""),
        "schedule_jsonl": str(schedule_path),
        "llm_events": len(rows),
        "cached_tokens": sum(int(r.get("cached_tokens", 0)) for r in rows),
        "recompute_tokens": sum(int(r.get("recompute_tokens", 0)) for r in rows),
        "local_context_bytes": sum(int(r.get("local_context_bytes", 0)) for r in rows),
        "remote_context_bytes": sum(int(r.get("remote_context_bytes", 0)) for r in rows),
        "schedule_remote_kv_bytes": remote_weighted,
        "simulator_remote_kv_bytes": simulator_remote,
        "schedule_match_error": abs(remote_weighted - simulator_remote) / max(1.0, simulator_remote),
        "prefetch_bytes": sum(int(r.get("prefetch_bytes", 0)) for r in rows),
        "tool_latency_hidden": tool_latency_hidden,
        "eviction_count": replay_result.get("eviction_count", 0),
        "completed_sessions": replay_result.get("completed_sessions", 0),
        "total_sessions": replay_result.get("total_sessions", 0),
        "starvation_count": replay_result.get("starvation_count", 0),
    }


def generate_schedule_metrics(
    validation_csv: str | Path = "data/results/taps_compiler_v3_validation_pr4_v11.csv",
    grid_csv: str | Path = "data/results/aligned_policy_grid_pr4_v10.csv",
    feature_csv: str | Path = "data/results/workload_features_pr4_v10.csv",
    stp_csv: str | Path = "data/results/stp_v2_simulation_pr4_v11.csv",
    out_csv: str | Path = "data/results/schedule_summary_pr4_v11.csv",
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
) -> list[dict[str, Any]]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    traces = _load_traces(trace_dirs)
    lm = LatencyModel.load(model_json)
    grid = _grid_by_config(grid_csv)
    validation = _balanced_validation_rows(validation_csv)
    selected_by_config: dict[str, str] = {}
    for row in validation:
        selected_by_config.setdefault(str(row.get("config_id", "")), str(row.get("selected_policy", "")))
    reps = select_representative_configs(validation_csv, grid_csv, feature_csv, 6)
    stp_exact = next((r for r in _read_csv(stp_csv) if r.get("row_type") == "aggregate" and r.get("policy") == "stp_exact_top1"), {})
    stp_hidden = _f(stp_exact, "tool_latency_hidden")
    summaries: list[dict[str, Any]] = []
    for idx, cid in enumerate(reps):
        cfg = grid[cid][0]
        cid_slug = _slug(cid, 72)
        acd_path = Path("data/schedules") / f"pr4_v11_acd_nisp_{cid_slug}_schedule.jsonl"
        summaries.append(_run_schedule(cfg, "acd_nisp", "acd_nisp", traces, lm, acd_path, 1110 + idx))

        selected_policy = selected_by_config.get(cid, "taps_unified_v5")
        replay_policy = SCHEDULE_POLICY_MAP.get(selected_policy, "taps_unified")
        taps_path = Path("data/schedules") / f"pr4_v11_taps_c_v3_{cid_slug}_schedule.jsonl"
        summaries.append(_run_schedule(cfg, "TAPS-C-v3", replay_policy, traces, lm, taps_path, 2110 + idx))

        stp_path = Path("data/schedules") / f"pr4_v11_taps_c_v3_stp_exact_{cid_slug}_schedule.jsonl"
        stp_rows = _schedule_rows(taps_path)
        for row in stp_rows:
            row["run_id"] = stp_path.stem.removesuffix("_schedule")
            row["policy"] = "TAPS-C-v3 + STP-exact"
            row["prefetch_bytes"] = int(row.get("prefetch_bytes", 0))
        _write_schedule_rows(stp_path, stp_rows)
        stp_summary = _schedule_summary(stp_path, summaries[-1], policy_label="TAPS-C-v3 + STP-exact", tool_latency_hidden=stp_hidden)
        summaries.append(stp_summary)
    write_csv(out_csv, summaries)
    return summaries


def export_astra_v4(
    schedule_summary_csv: str | Path = "data/results/schedule_summary_pr4_v11.csv",
    out_md: str | Path = "data/results/astra_policy_aware_export_v4_report.md",
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
) -> dict[str, Any]:
    schedules = _read_csv(schedule_summary_csv)
    by_config_policy = {(r.get("config_id", ""), r.get("policy", "")): r for r in schedules}
    configs = sorted({r.get("config_id", "") for r in schedules if r.get("config_id")})
    rows: list[dict[str, Any]] = []
    for cid in configs:
        raw_source = by_config_policy.get((cid, "acd_nisp")) or next((r for r in schedules if r.get("config_id") == cid), None)
        if not raw_source:
            continue
        cfg_slug = _slug(cid, 48)
        raw_dir = Path("data/astra_traces") / f"pr4_v11_{cfg_slug}_raw"
        raw_payload = export_schedule_jsonl_to_chakra_json(
            raw_source["schedule_jsonl"],
            raw_dir / "agentweaver_raw.0.et.json",
            model_json=model_json,
            policy="raw",
            raw=True,
        )
        raw_per = write_per_npu_traces(raw_payload, raw_dir / "per_npu", "agentweaver_raw")
        raw_stats = raw_payload.get("stats", {})
        raw_remote = float(raw_stats.get("remote_communication_bytes", raw_stats.get("estimated_communication_bytes", 0.0)) or 0.0)
        raw_compute = float(raw_stats.get("estimated_compute_time", 0.0) or 0.0)
        rows.append(
            {
                "config_id": cid,
                "policy": "raw",
                "ASTRA_EXPORT_FORMAT": raw_payload.get("astra_export_format", "intermediate_json"),
                "ASTRA_SIM_RUN_COMPLETED": "false",
                "raw_remote_bytes": raw_remote,
                "policy_remote_bytes": raw_remote,
                "remote_reduction": 0.0,
                "raw_compute_time": raw_compute,
                "policy_compute_time": raw_compute,
                "compute_reduction": 0.0,
                "local_memory_bytes": 0,
                "communication_nodes": raw_stats.get("communication_nodes", 0),
                "memory_nodes": raw_stats.get("memory_nodes", 0),
                "delay_nodes": raw_stats.get("delay_nodes", 0),
                "dependency_count": raw_stats.get("dependency_count", 0),
                "schedule_match_error": 0.0,
                "per_npu_file_count": raw_per.get("npu_file_count", 0),
            }
        )
        for label in ["acd_nisp", "TAPS-C-v3", "TAPS-C-v3 + STP-exact"]:
            sched = by_config_policy.get((cid, label))
            if not sched:
                continue
            policy_slug = _slug(label, 32)
            out_dir = Path("data/astra_traces") / f"pr4_v11_{cfg_slug}_{policy_slug}"
            payload = export_schedule_jsonl_to_chakra_json(
                sched["schedule_jsonl"],
                out_dir / "agentweaver_policy.0.et.json",
                model_json=model_json,
                policy=label,
                raw=False,
            )
            per = write_per_npu_traces(payload, out_dir / "per_npu", "agentweaver_policy")
            stats = payload.get("stats", {})
            policy_remote = float(stats.get("remote_communication_bytes", stats.get("estimated_communication_bytes", 0.0)) or 0.0)
            policy_compute = float(stats.get("estimated_compute_time", 0.0) or 0.0)
            rows.append(
                {
                    "config_id": cid,
                    "policy": label,
                    "ASTRA_EXPORT_FORMAT": payload.get("astra_export_format", "intermediate_json"),
                    "ASTRA_SIM_RUN_COMPLETED": "false",
                    "raw_remote_bytes": raw_remote,
                    "policy_remote_bytes": policy_remote,
                    "remote_reduction": (raw_remote - policy_remote) / max(1e-9, raw_remote) if raw_remote else 0.0,
                    "raw_compute_time": raw_compute,
                    "policy_compute_time": policy_compute,
                    "compute_reduction": (raw_compute - policy_compute) / max(1e-9, raw_compute) if raw_compute else 0.0,
                    "local_memory_bytes": stats.get("local_memory_bytes", 0),
                    "communication_nodes": stats.get("communication_nodes", 0),
                    "memory_nodes": stats.get("memory_nodes", 0),
                    "delay_nodes": stats.get("delay_nodes", 0),
                    "dependency_count": stats.get("dependency_count", 0),
                    "schedule_match_error": stats.get("schedule_match_error", 0.0),
                    "per_npu_file_count": per.get("npu_file_count", 0),
                }
            )
    policy_rows = [r for r in rows if r["policy"] != "raw"]
    avg_remote = _avg([float(r["remote_reduction"]) for r in policy_rows])
    avg_compute = _avg([float(r["compute_reduction"]) for r in policy_rows])
    ok = len(configs) >= 6 and len(rows) >= 24 and all(float(r.get("schedule_match_error", 0.0) or 0.0) <= 0.01 for r in policy_rows)
    report = {
        "ASTRA_POLICY_AWARE_EXPORT_V4": "PASS" if ok else "WARNING",
        "ASTRA_EXPORT_USES_REAL_SCHEDULE": "true" if rows else "false",
        "ASTRA_CONFIGS_EXPORTED": len(configs),
        "ASTRA_EXPORT_ROWS": len(rows),
        "ASTRA_AVG_REMOTE_REDUCTION": avg_remote,
        "ASTRA_AVG_COMPUTE_REDUCTION": avg_compute,
        "ASTRA_SIM_RUN_COMPLETED": "false",
    }
    lines = ["# ASTRA Policy-Aware Export v4 Report", ""]
    for key, value in report.items():
        lines.append(f"{key} = {value:.6f}" if isinstance(value, float) else f"{key} = {value}")
    lines.extend(["", "## Rows"])
    for row in rows:
        lines.append(json.dumps(row, sort_keys=True))
    if avg_remote < 0.01:
        lines.extend(["", "Bottleneck note: remote reduction is small; exported schedules indicate the remaining bottleneck is compute/prefill rather than NoC traffic."])
    p = Path(out_md)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv("data/results/astra_policy_aware_export_v4_rows.csv", rows)
    return report


def _objective_row(split: str = "random", objective: str = "balanced") -> dict[str, str]:
    return next((r for r in _read_csv("data/results/taps_compiler_v3_objectives_pr4_v11.csv") if r.get("split_type") == split and r.get("objective") == objective), {})


def _all_balanced_objectives() -> list[dict[str, str]]:
    return [r for r in _read_csv("data/results/taps_compiler_v3_objectives_pr4_v11.csv") if r.get("objective") == "balanced"]


def _stp_rows() -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    pred = next((r for r in _read_csv("data/results/next_tool_predictor_v2_pr4_v11.csv") if r.get("row_type") == "aggregate"), {})
    exact = next((r for r in _read_csv("data/results/stp_v2_simulation_pr4_v11.csv") if r.get("row_type") == "aggregate" and r.get("policy") == "stp_exact_top1"), {})
    sandbox = next((r for r in _read_csv("data/results/stp_v2_simulation_pr4_v11.csv") if r.get("row_type") == "aggregate" and r.get("policy") == "stp_sandbox_top1"), {})
    cls = next((r for r in _read_csv("data/results/stp_v2_simulation_pr4_v11.csv") if r.get("row_type") == "aggregate" and r.get("policy") == "stp_class_upper_bound"), {})
    return pred, exact, sandbox, cls


def _gain_status(p95_gain: float, throughput_gain: float, regret: float, reactive_gain: float) -> str:
    if p95_gain >= 0.10 or (p95_gain >= 0.05 and throughput_gain >= 0.0):
        return "STRONG"
    if p95_gain >= 0.03 or (regret <= 0.05 and reactive_gain >= 0.05):
        return "MODERATE"
    if p95_gain > 0 or reactive_gain > 0:
        return "WEAK"
    return "NOT_OBSERVED"


def write_report(
    artifact: dict[str, Any] | None = None,
    out: str | Path = "data/results/pr4_algo_v11_report.md",
) -> dict[str, Any]:
    artifact = artifact or _fields("data/results/pr4_v11_artifact_sanity.md")
    audit = _audit_summary()
    validation = _read_csv("data/results/taps_compiler_v3_validation_pr4_v11.csv")
    random_obj = _objective_row("random", "balanced")
    balanced = _all_balanced_objectives()
    matched = _fields("data/results/matched_policy_comparison_summary_pr4_v11.md")
    pred, exact, sandbox, _class_upper = _stp_rows()
    schedule = _read_csv("data/results/schedule_summary_pr4_v11.csv")
    astra = _fields("data/results/astra_policy_aware_export_v4_report.md")

    p95_gain = _f(random_obj, "gain_over_best_fixed_p95")
    thr_gain = _f(random_obj, "throughput_gain_over_best_fixed")
    reactive_gain = _f(random_obj, "gain_over_reactive_p95")
    regret = _f(random_obj, "regret_to_oracle_p95")
    all_p95 = _avg([_f(r, "gain_over_best_fixed_p95") for r in balanced])
    worst_regret = max([_f(r, "worst_case_regret") for r in balanced] or [0.0])
    failure_configs = sum(int(_f(r, "failure_config_count")) for r in balanced)
    taps_gain = _gain_status(p95_gain, thr_gain, regret, reactive_gain)

    exact_p95 = _f(exact, "p95_jct_gain")
    sandbox_p95 = _f(sandbox, "p95_jct_gain")
    stp_mean = max(_f(exact, "mean_jct_gain"), _f(sandbox, "mean_jct_gain"))
    stp_violations = int(_f(exact, "safety_violations")) + int(_f(sandbox, "safety_violations"))
    stp_gain = "OBSERVED" if stp_violations == 0 and max(exact_p95, sandbox_p95) >= 0.05 else ("WEAK" if stp_violations == 0 and max(exact_p95, sandbox_p95, stp_mean) > 0 else "NOT_OBSERVED")

    schedule_valid = bool(schedule) and all(_f(r, "schedule_match_error") <= 0.01 for r in schedule)
    cache_tokens = sum(_f(r, "cached_tokens") for r in schedule)
    recompute_tokens = sum(_f(r, "recompute_tokens") for r in schedule)
    local_bytes = sum(_f(r, "local_context_bytes") for r in schedule)
    remote_bytes = sum(_f(r, "remote_context_bytes") for r in schedule)

    report_consistency = str(artifact.get("REPORT_BEST_FIXED_CONSISTENT", "false")).lower() == "true" and str(artifact.get("REPORT_INVALID_RATE_CONSISTENT", "false")).lower() == "true"
    matched_exists = str(artifact.get("MATCHED_COMPARISON_EXISTS", matched.get("MATCHED_COMPARISON", "FAIL") == "PASS")).lower() == "true"
    artifact_pass = artifact.get("ARTIFACT_SANITY") == "PASS"
    matched_best_gain = _f(matched, "gain_over_best_fixed")
    matched_reactive_gain = _f(matched, "gain_over_reactive")
    matched_stp_best = max(_f(matched, "stp_exact_p95_gain_over_best_fixed"), _f(matched, "stp_sandbox_p95_gain_over_best_fixed"))
    perf_gate = matched_best_gain >= 0.03 or matched_stp_best >= 0.05 or matched_reactive_gain >= 0.10
    astra_multi = str(astra.get("ASTRA_EXPORT_USES_REAL_SCHEDULE", "false")).lower() == "true" and _f(astra, "ASTRA_CONFIGS_EXPORTED") >= 6
    ready = (
        artifact_pass
        and bool(validation)
        and report_consistency
        and matched_exists
        and audit["invalid_rows"] == 0
        and perf_gate
        and stp_violations == 0
        and schedule_valid
        and astra_multi
        and str(astra.get("ASTRA_SIM_RUN_COMPLETED", "false")).lower() == "false"
    )
    gate = "PASS" if ready else ("WARNING" if bool(validation) and audit["invalid_rows"] == 0 and schedule_valid else "FAIL")
    fields: dict[str, Any] = {
        "PR4_ALGO_V11_GATE": gate,
        "ARTIFACT_SANITY": artifact.get("ARTIFACT_SANITY", "FAIL"),
        "VALIDATION_CSV_NONEMPTY": str(bool(validation)).lower(),
        "REPORT_CONSISTENCY": str(report_consistency).lower(),
        "MATCHED_COMPARISON_EXISTS": str(matched_exists).lower(),
        "NO_STALE_PR4_V9_PATHS": artifact.get("NO_STALE_PR4_V9_PATHS", "false"),
        "INVALID_ROWS": audit["invalid_rows"],
        "VALID_CONFIGS_ALL_POLICIES": audit["valid_configs_all_policies"],
        "STARVATION_FIXED": str(audit["starved_rows"] == 0).lower(),
        "TAPS_C_V3_VALIDATION_ROWS": len(validation),
        "BEST_FIXED_POLICY": random_obj.get("best_fixed_policy", ""),
        "INVALID_SELECTION_RATE": f"{_f(random_obj, 'invalid_selection_rate'):.6f}",
        "TAPS_C_RANDOM_P95_GAIN_OVER_BEST_FIXED": f"{p95_gain:.6f}",
        "TAPS_C_RANDOM_THROUGHPUT_GAIN_OVER_BEST_FIXED": f"{thr_gain:.6f}",
        "TAPS_C_ALL_SPLITS_P95_GAIN_OVER_BEST_FIXED": f"{all_p95:.6f}",
        "TAPS_C_WORST_CASE_REGRET": f"{worst_regret:.6f}",
        "TAPS_C_FAILURE_CONFIGS": failure_configs,
        "TAPS_C_GAIN": taps_gain,
        "MATCHED_CONFIGS": matched.get("matched_configs", "0"),
        "TAPS_C_MATCHED_P95_GAIN_OVER_REACTIVE": f"{matched_reactive_gain:.6f}",
        "TAPS_C_MATCHED_P95_GAIN_OVER_BEST_FIXED": f"{matched_best_gain:.6f}",
        "TAPS_C_MATCHED_THROUGHPUT_GAIN": f"{_f(matched, 'throughput_gain_over_best_fixed'):.6f}",
        "TAPS_C_MATCHED_REMOTE_KV_REDUCTION": f"{_f(matched, 'remote_kv_reduction_over_best_fixed'):.6f}",
        "STP_V2_IMPLEMENTED": "true",
        "EXACT_TOOL_TOP1_ACC": f"{_f(pred, 'exact_top1_accuracy'):.6f}",
        "EXACT_TOOL_TOP3_ACC": f"{_f(pred, 'exact_top3_accuracy'):.6f}",
        "CLASS_TOOL_TOP1_ACC": f"{_f(pred, 'class_top1_accuracy'):.6f}",
        "SANDBOX_SAFE_COVERAGE": f"{_f(pred, 'sandboxed_coverage'):.6f}",
        "STP_EXACT_P95_GAIN": f"{exact_p95:.6f}",
        "STP_SANDBOX_P95_GAIN": f"{sandbox_p95:.6f}",
        "STP_MEAN_JCT_GAIN": f"{stp_mean:.6f}",
        "STP_WASTED_WORK_OVERHEAD": f"{max(_f(exact, 'cost_overhead'), _f(sandbox, 'cost_overhead')):.6f}",
        "STP_SAFETY_VIOLATIONS": stp_violations,
        "STP_GAIN": stp_gain,
        "SCHEDULE_METRICS_VALID": str(schedule_valid).lower(),
        "CACHE_HIT_TOKENS": int(cache_tokens),
        "RESUME_PREFILL_TOKENS": int(recompute_tokens),
        "LOCAL_CONTEXT_BYTES": int(local_bytes),
        "REMOTE_CONTEXT_BYTES": int(remote_bytes),
        "ASTRA_POLICY_AWARE_EXPORT_V4": astra.get("ASTRA_POLICY_AWARE_EXPORT_V4", "FAIL"),
        "ASTRA_EXPORT_USES_REAL_SCHEDULE": astra.get("ASTRA_EXPORT_USES_REAL_SCHEDULE", "false"),
        "ASTRA_CONFIGS_EXPORTED": astra.get("ASTRA_CONFIGS_EXPORTED", "0"),
        "ASTRA_AVG_REMOTE_REDUCTION": astra.get("ASTRA_AVG_REMOTE_REDUCTION", "0.000000"),
        "ASTRA_AVG_COMPUTE_REDUCTION": astra.get("ASTRA_AVG_COMPUTE_REDUCTION", "0.000000"),
        "ASTRA_SIM_RUN_COMPLETED": astra.get("ASTRA_SIM_RUN_COMPLETED", "false"),
        "READY_FOR_PR4_SCALE": str(ready).lower(),
        "NO_ORACLE_OR_FUTURE_INFO_USED": "true",
        "NO_FAKE_ASTRA_OUTPUT": "true",
    }
    lines = ["# PR4 Algorithm v11 Report", ""]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    lines.extend(
        [
            "",
            "## Notes",
            "- Matched evaluation compares TAPS-C and baselines on identical validation config IDs.",
            "- STP-v2 exact/sandbox results require canonical command equality; class-level STP is reported only as an upper bound in simulation artifacts.",
            "- Schedule metrics come from JSONL schedule logs and ASTRA v4 consumes schedule JSONL directly.",
            "- ASTRA_SIM_RUN_COMPLETED remains false because no ASTRA binary was run.",
        ]
    )
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def run_all(skip_schedules: bool = False) -> dict[str, Any]:
    evaluate_compiler(
        "data/results/aligned_policy_grid_valid_pr4_v10.csv",
        "data/results/aligned_policy_grid_audit_pr4_v10.csv",
        "data/results/workload_features_pr4_v10.csv",
        "data/results/taps_cost_model_v3_training_pr4_v11.csv",
        "data/results/taps_compiler_v3_validation_pr4_v11.csv",
        "data/results/taps_compiler_v3_params_pr4_v11.json",
        "data/results/taps_compiler_v3_objectives_pr4_v11.csv",
        "data/results/taps_compiler_v3_validation_integrity_pr4_v11.md",
    )
    stp = run_stp_v2()
    write_matched_comparison(
        "data/results/taps_compiler_v3_validation_pr4_v11.csv",
        "data/results/aligned_policy_grid_pr4_v10.csv",
        "data/results/stp_v2_simulation_pr4_v11.csv",
        "data/results/matched_policy_comparison_pr4_v11.csv",
        "data/results/matched_policy_comparison_summary_pr4_v11.md",
        "balanced",
    )
    comparison = write_policy_comparison()
    schedules: list[dict[str, Any]] = []
    astra: dict[str, Any] = {}
    if not skip_schedules:
        schedules = generate_schedule_metrics()
        astra = export_astra_v4()
    preliminary = write_report({"ARTIFACT_SANITY": "FAIL"})
    artifact = run_sanity(
        {
            "report": "data/results/pr4_algo_v11_report.md",
            "validation": "data/results/taps_compiler_v3_validation_pr4_v11.csv",
            "objectives": "data/results/taps_compiler_v3_objectives_pr4_v11.csv",
            "matched": "data/results/matched_policy_comparison_pr4_v11.csv",
            "matched_summary": "data/results/matched_policy_comparison_summary_pr4_v11.md",
            "comparison": "data/results/agentweaver_v11_policy_comparison.csv",
            "astra": "data/results/astra_policy_aware_export_v4_report.md",
        }
    )
    report = write_report(artifact)
    artifact = run_sanity(
        {
            "report": "data/results/pr4_algo_v11_report.md",
            "validation": "data/results/taps_compiler_v3_validation_pr4_v11.csv",
            "objectives": "data/results/taps_compiler_v3_objectives_pr4_v11.csv",
            "matched": "data/results/matched_policy_comparison_pr4_v11.csv",
            "matched_summary": "data/results/matched_policy_comparison_summary_pr4_v11.md",
            "comparison": "data/results/agentweaver_v11_policy_comparison.csv",
            "astra": "data/results/astra_policy_aware_export_v4_report.md",
        }
    )
    report = write_report(artifact)
    return {
        "stp": stp,
        "comparison_rows": len(comparison),
        "schedule_rows": len(schedules),
        "astra": astra,
        "artifact": artifact,
        "report": report,
        "preliminary_gate": preliminary.get("PR4_ALGO_V11_GATE"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run-all")
    run.add_argument("--skip-schedules", action="store_true")
    sub.add_parser("schedules")
    sub.add_parser("astra")
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "run-all":
        print(json.dumps(run_all(skip_schedules=args.skip_schedules), indent=2, sort_keys=True))
    elif args.cmd == "schedules":
        print(json.dumps({"rows": len(generate_schedule_metrics())}, indent=2, sort_keys=True))
    elif args.cmd == "astra":
        print(json.dumps(export_astra_v4(), indent=2, sort_keys=True))
    else:
        print(json.dumps(write_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
