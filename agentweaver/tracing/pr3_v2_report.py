from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from agentweaver.profiling.pr2_v2 import REQUIRED_REAL_POLICIES
from agentweaver.utils.io import ensure_dir


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    return int(_float(row, key, float(default)))


def _aggregate(path: str | Path) -> dict[str, str]:
    rows = _read_csv(path)
    for row in rows:
        if row.get("trace") == "AGGREGATE":
            return row
    return rows[-1] if rows else {}


def _instance_count(path: str | Path) -> int:
    return len({r.get("instance_id") for r in _read_csv(path) if r.get("instance_id")})


def _llm_match_rate(validation_path: str | Path) -> float:
    row = _aggregate(validation_path)
    total = _int(row, "num_llm_events")
    matched = _int(row, "llm_timing_matched")
    return matched / total if total else 0.0


def _tool_timing_available(validation_path: str | Path) -> bool:
    row = _aggregate(validation_path)
    return _int(row, "tool_timing_available") > 0


def _measured_jct_available(validation_path: str | Path, summary_path: str | Path) -> bool:
    row = _aggregate(validation_path)
    if str(row.get("measured_agent_jct_available", "")).lower() == "true":
        return True
    return any(_float(r, "measured_agent_jct") > 0 for r in _read_csv(summary_path))


def _all_policies(path: str | Path) -> bool:
    policies = {r.get("policy") for r in _read_csv(path) if r.get("instance_id") == "AGGREGATE"}
    return set(REQUIRED_REAL_POLICIES).issubset(policies)


def _complete_rollout_instances(branch_summary: str | Path, rollouts: int = 4) -> int:
    by_instance: dict[str, set[str]] = {}
    for row in _read_csv(branch_summary):
        instance = row.get("instance_id") or ""
        branch = row.get("branch_id") or ""
        if instance and branch.startswith("b"):
            by_instance.setdefault(instance, set()).add(branch)
    required = {f"b{i}" for i in range(rollouts)}
    return sum(1 for branches in by_instance.values() if required.issubset(branches))


def _branch_skew(branch_summary: str | Path) -> bool:
    for row in _read_csv(branch_summary):
        if row.get("branch_id") != "AGGREGATE":
            continue
        if (
            _float(row, "branch_jct_cv") > 0
            or _float(row, "branch_event_count_std") > 0
            or _float(row, "llm_input_tokens_std") > 0
            or _float(row, "llm_output_tokens_std") > 0
            or str(row.get("branches_have_different_lengths", "")).lower() == "true"
        ):
            return True
    return False


def _shared_context(paths: list[Path]) -> bool:
    for path in paths:
        for row in _read_csv(path):
            if _float(row, "repeated_prefill_tokens") > 0 or _float(row, "exact_prefix_reusable_tokens") > 0:
                return True
    return False


def _unknown_verifiers(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        total += sum(_int(row, "unknown_verifier_results") for row in _read_csv(path))
    return total


def _official_status(results: Path) -> tuple[bool, int, str]:
    summary = results / "mini_swe_lite5_timed_official_eval_summary.csv"
    rows = _read_csv(summary)
    used = any(str(r.get("official_verifier_used", "")).lower() == "true" for r in rows)
    if used:
        predictions = results / "mini_swe_lite5_timed_predictions.jsonl"
        try:
            evaluated = len([line for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()])
        except Exception:
            evaluated = 0
        return True, evaluated, "official harness completed"
    if rows:
        return False, 0, rows[0].get("message", "official harness unavailable or skipped")
    return False, 0, "official harness not run"


def _bes_effect(results: Path) -> str:
    rows = {
        r.get("policy", ""): r
        for r in _read_csv(results / "mini_swe_lite10_r4_timed_policy_comparison.csv")
        if r.get("policy")
    }
    acd = rows.get("acd_only")
    bes = rows.get("acd_bes")
    if not acd or not bes:
        return "UNKNOWN"
    keys = ("jct", "prefill_tokens_avoided", "resume_prefill_tokens", "region_utilization")
    same = all(abs(_float(acd, k) - _float(bes, k)) < 1e-9 for k in keys)
    return "WARNING_ACD_BES_EQUALS_ACD_ONLY" if same else "OBSERVED"


def write_report(results_dir: str | Path = "data/results", out: str | Path = "data/results/pr3_v2_report.md") -> dict[str, str]:
    results = Path(results_dir)
    lite5_validation = results / "mini_swe_lite5_timed_trace_validation.csv"
    lite5_summary = results / "mini_swe_lite5_timed_trace_summary.csv"
    lite5_replay = results / "mini_swe_lite5_timed_replay_all_policies.csv"
    lite10_validation = results / "mini_swe_lite10_r4_timed_trace_validation.csv"
    lite10_summary = results / "mini_swe_lite10_r4_timed_trace_summary.csv"
    lite10_branch = results / "mini_swe_lite10_r4_timed_branch_summary.csv"
    lite10_replay = results / "mini_swe_lite10_r4_timed_replay_all_policies.csv"

    lite5_match = _llm_match_rate(lite5_validation)
    lite10_match = _llm_match_rate(lite10_validation)
    tool_timing = _tool_timing_available(lite5_validation) or _tool_timing_available(lite10_validation)
    measured_jct = _measured_jct_available(lite5_validation, lite5_summary) or _measured_jct_available(lite10_validation, lite10_summary)
    lite5_instances = _instance_count(lite5_summary)
    lite10_complete = _complete_rollout_instances(lite10_branch, 4)
    lite10_instances = _instance_count(lite10_summary)
    lite5_policies = _all_policies(lite5_replay)
    lite10_policies = _all_policies(lite10_replay)

    lite5_status = "PASS" if lite5_instances >= 5 and lite5_match >= 0.8 and measured_jct and lite5_policies else ("WARNING" if lite5_instances > 0 else "FAIL")
    lite10_status = (
        "PASS"
        if lite10_complete >= 5 and lite10_match >= 0.8 and measured_jct and lite10_policies
        else ("WARNING" if lite10_instances > 0 else "FAIL")
    )
    official_used, official_n, official_note = _official_status(results)
    bes_effect = _bes_effect(results)
    unknown = _unknown_verifiers([lite5_summary, lite10_summary])
    shared = _shared_context(
        [
            results / "mini_swe_lite5_timed_context_reuse.csv",
            results / "mini_swe_lite10_r4_timed_context_reuse.csv",
        ]
    )
    skew = _branch_skew(lite10_branch)
    tool_stall = _int(_aggregate(lite5_validation), "num_tool_events") + _int(_aggregate(lite10_validation), "num_tool_events") > 0
    plots = [
        Path("data/plots/mini_swe_lite5_timed_latency_breakdown.pdf"),
        Path("data/plots/mini_swe_lite10_r4_timed_branch_jct_cdf.pdf"),
        Path("data/plots/mini_swe_lite10_r4_timed_tool_latency_cdf.pdf"),
        Path("data/plots/mini_swe_lite10_r4_timed_policy_comparison.pdf"),
        Path("data/plots/mini_swe_lite10_r4_timed_context_reuse.pdf"),
    ]
    plots_exist = all(p.exists() for p in plots)
    official_marked = (results / "mini_swe_lite5_timed_official_eval_summary.csv").exists()
    ready = (
        lite5_status == "PASS"
        and lite10_status != "FAIL"
        and lite5_match >= 0.8
        and lite10_match >= 0.8
        and lite5_policies
        and lite10_policies
        and measured_jct
        and plots_exist
        and (official_used or official_marked)
    )
    if lite5_status == "FAIL" or lite10_status == "FAIL" or not lite5_policies or not lite10_policies:
        gate = "FAIL"
    elif not official_used:
        gate = "WARNING"
    else:
        gate = "PASS"
    if tool_timing:
        instrumentation = "PATCHED_MINISWE"
    elif lite5_match >= 0.8 or lite10_match >= 0.8:
        instrumentation = "VLLM_PROXY"
    else:
        instrumentation = "FAILED"

    fields = {
        "PR3_V2_GATE": gate,
        "TIMING_INSTRUMENTATION": instrumentation,
        "LLM_TIMING_MATCH_RATE_LITE5": f"{lite5_match:.4f}",
        "LLM_TIMING_MATCH_RATE_LITE10_R4": f"{lite10_match:.4f}",
        "TOOL_TIMING_AVAILABLE": str(tool_timing).lower(),
        "MEASURED_AGENT_JCT_AVAILABLE": str(measured_jct).lower(),
        "MINI_SWE_LITE5_TIMED": lite5_status,
        "MINI_SWE_LITE10_R4_TIMED": lite10_status,
        "OFFICIAL_VERIFIER_USED": str(official_used).lower(),
        "OFFICIAL_VERIFIER_NUM_EVALUATED": str(official_n),
        "UNKNOWN_VERIFIER_RESULTS": str(unknown),
        "SHARED_CONTEXT_REUSE_OBSERVED": str(shared).lower(),
        "BRANCH_SKEW_OBSERVED": str(skew).lower(),
        "TOOL_STALL_RESUME_OBSERVED": str(tool_stall).lower(),
        "ALL_POLICY_REPLAY_TIMED_LITE5": "PASS" if lite5_policies else "FAIL",
        "ALL_POLICY_REPLAY_TIMED_LITE10_R4": "PASS" if lite10_policies else "FAIL",
        "BES_REAL_TRACE_EFFECT": bes_effect,
        "READY_FOR_PR4_SCALEUP": str(ready).lower(),
    }
    notes = [
        "No timestamps, tool latency, or verifier pass/fail are fabricated.",
        f"Official verifier status: {official_note}.",
        "Solved rate is not reported unless official SWE-bench harness completes.",
        "BES remains identical to ACD-only on this 6x6/4-branch timed trace if BES_REAL_TRACE_EFFECT is WARNING; use PR4 scale/constrained-resource runs to isolate BES scheduling pressure.",
    ]
    outp = Path(out)
    ensure_dir(outp.parent)
    lines = ["# PR3-v2 Report", ""]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    lines.append("")
    lines.append("## Notes")
    lines.extend(f"- {note}" for note in notes)
    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="data/results")
    ap.add_argument("--out", default="data/results/pr3_v2_report.md")
    args = ap.parse_args()
    fields = write_report(args.results_dir, args.out)
    for key, value in fields.items():
        print(f"{key} = {value}")


if __name__ == "__main__":
    main()
