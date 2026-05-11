from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from agentweaver.utils.io import ensure_dir


DEFAULT_FILES = {
    "report": "data/results/pr4_algo_v10_report.md",
    "grid": "data/results/aligned_policy_grid_pr4_v10.csv",
    "audit_md": "data/results/aligned_policy_grid_audit_pr4_v10.md",
    "audit_csv": "data/results/aligned_policy_grid_audit_pr4_v10.csv",
    "validation": "data/results/taps_compiler_v3_validation_pr4_v11.csv",
    "objectives": "data/results/taps_compiler_v3_objectives_pr4_v11.csv",
    "comparison": "data/results/agentweaver_v11_policy_comparison.csv",
    "matched": "data/results/matched_policy_comparison_pr4_v11.csv",
    "matched_summary": "data/results/matched_policy_comparison_summary_pr4_v11.md",
    "stp": "data/results/stp_v2_simulation_pr4_v11.csv",
    "astra": "data/results/astra_policy_aware_export_v4_report.md",
}

SOURCE_FILES_TO_SCAN = [
    "agentweaver/analysis/matched_policy_comparison.py",
    "agentweaver/analysis/pr4_algo_v11.py",
    "agentweaver/simulator/taps_cost_model_v3.py",
    "agentweaver/simulator/safe_tool_prefetch_v2.py",
    "agentweaver/simulator/agentweaver_v11_replay.py",
]


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
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        key = key.strip()
        if re.fullmatch(r"[A-Za-z0-9_]+", key):
            out[key] = value.strip()
    return out


def _f(row: dict[str, Any] | None, key: str, default: float = math.nan) -> float:
    if not row:
        return default
    try:
        value = row.get(key)
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / max(1, len(vals))


def _bool_field(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _validation_summary(validation: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    keys = sorted({(r.get("split_type", ""), r.get("objective", "")) for r in validation})
    for split, objective in keys:
        rows = [r for r in validation if r.get("split_type") == split and r.get("objective") == objective]
        if not rows:
            continue
        invalid = sum(1 for r in rows if _bool_field(r.get("invalid_selection")))
        fallback = sum(1 for r in rows if _bool_field(r.get("fallback_used")))
        failures = {r.get("config_id", "") for r in rows if _f(r, "gain_over_best_fixed_p95", 0.0) < 0}
        best_counts: dict[str, int] = {}
        selected_counts: dict[str, int] = {}
        for r in rows:
            best_counts[str(r.get("best_fixed_policy", ""))] = best_counts.get(str(r.get("best_fixed_policy", "")), 0) + 1
            selected_counts[str(r.get("selected_policy", ""))] = selected_counts.get(str(r.get("selected_policy", "")), 0) + 1
        best_fixed = max(best_counts, key=best_counts.get) if best_counts else ""
        out[(split, objective)] = {
            "validation_rows": len(rows),
            "best_fixed_policy": best_fixed,
            "selected_policy_distribution": selected_counts,
            "invalid_selection_rate": invalid / max(1, len(rows)),
            "fallback_rate": fallback / max(1, len(rows)),
            "gain_over_best_fixed_p95": _avg(rows, "gain_over_best_fixed_p95"),
            "throughput_gain_over_best_fixed": _avg(rows, "throughput_gain_over_best_fixed"),
            "gain_over_reactive_p95": _avg(rows, "gain_over_reactive_p95"),
            "regret_to_oracle_p95": _avg(rows, "regret_to_oracle_p95"),
            "worst_case_regret": max([_f(r, "regret_to_oracle_p95", 0.0) for r in rows] or [0.0]),
            "failure_config_count": len(failures),
        }
    return out


def _objectives_reproducible(validation: list[dict[str, str]], objectives: list[dict[str, str]], tolerance: float = 1e-6) -> tuple[bool, list[str]]:
    if not validation or not objectives:
        return False, ["validation_or_objectives_empty"]
    summary = _validation_summary(validation)
    mismatches: list[str] = []
    numeric_keys = [
        "validation_rows",
        "invalid_selection_rate",
        "fallback_rate",
        "gain_over_best_fixed_p95",
        "throughput_gain_over_best_fixed",
        "gain_over_reactive_p95",
        "regret_to_oracle_p95",
        "worst_case_regret",
        "failure_config_count",
    ]
    for obj in objectives:
        key = (str(obj.get("split_type", "")), str(obj.get("objective", "")))
        got = summary.get(key)
        if not got:
            mismatches.append(f"{key}:missing_from_validation")
            continue
        if str(obj.get("best_fixed_policy", "")) != str(got.get("best_fixed_policy", "")):
            mismatches.append(f"{key}:best_fixed:{obj.get('best_fixed_policy')}!={got.get('best_fixed_policy')}")
        for nkey in numeric_keys:
            ov = _f(obj, nkey)
            gv = float(got.get(nkey, math.nan))
            if not (math.isfinite(ov) and math.isfinite(gv)) or abs(ov - gv) > tolerance:
                mismatches.append(f"{key}:{nkey}:{ov}!={gv}")
                break
    return not mismatches, mismatches


def _report_best_fixed_consistent(report: dict[str, str], validation: list[dict[str, str]], objectives: list[dict[str, str]]) -> bool:
    if not report:
        return False
    report_best = report.get("BEST_FIXED_POLICY") or report.get("TAPS_C_BEST_FIXED_POLICY")
    objective_best = next(
        (r.get("best_fixed_policy", "") for r in objectives if r.get("split_type") == "random" and r.get("objective") == "balanced"),
        "",
    )
    validation_best = next(
        (r.get("best_fixed_policy", "") for r in validation if r.get("split_type") == "random" and r.get("objective") == "balanced"),
        "",
    )
    return bool(report_best and objective_best and validation_best and report_best == objective_best == validation_best)


def _report_invalid_rate_consistent(report: dict[str, str], validation: list[dict[str, str]], objectives: list[dict[str, str]]) -> bool:
    if not report:
        return False
    report_rate = _f(report, "INVALID_SELECTION_RATE")
    if not math.isfinite(report_rate):
        report_rate = _f(report, "TAPS_C_INVALID_SELECTION_RATE")
    objective_rate = next(
        (_f(r, "invalid_selection_rate") for r in objectives if r.get("split_type") == "random" and r.get("objective") == "balanced"),
        math.nan,
    )
    rows = [r for r in validation if r.get("split_type") == "random" and r.get("objective") == "balanced"]
    validation_rate = sum(1 for r in rows if _bool_field(r.get("invalid_selection"))) / max(1, len(rows)) if rows else math.nan
    vals = [report_rate, objective_rate, validation_rate]
    return all(math.isfinite(v) for v in vals) and max(vals) - min(vals) <= 1e-6


def _no_stale_v9_paths(files: dict[str, str]) -> tuple[bool, list[str]]:
    stale_path = re.compile(r"(data/results/[^\"'\s]*pr4_v9|pr4_algo_v9_report|aligned_policy[^\"'\s]*pr4_v9|taps_compiler[^\"'\s]*pr4_v9)")
    offenders: list[str] = []
    for key, value in files.items():
        if stale_path.search(str(value)):
            offenders.append(f"default:{key}:{value}")
    for source in SOURCE_FILES_TO_SCAN:
        p = Path(source)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        if stale_path.search(text):
            offenders.append(source)
    return not offenders, offenders


def _audit_invalid_rows(audit_rows: list[dict[str, str]]) -> int:
    if not audit_rows:
        return -1
    invalid = 0
    for row in audit_rows:
        valid = row.get("validity")
        if valid is None:
            valid = row.get("valid")
        if str(valid).lower() not in {"true", "1", "yes"}:
            invalid += 1
    return invalid


def _ready_gate_trustworthy(
    report: dict[str, str],
    *,
    artifact_core_ok: bool,
    matched_exists: bool,
    astra_ok: bool,
    invalid_rows: int,
) -> bool:
    ready = _bool_field(report.get("READY_FOR_PR4_SCALE", "false"))
    if not ready:
        return True
    return artifact_core_ok and matched_exists and astra_ok and invalid_rows == 0


def run_sanity(
    files: dict[str, str] | None = None,
    out_md: str | Path = "data/results/pr4_v11_artifact_sanity.md",
) -> dict[str, Any]:
    files = {**DEFAULT_FILES, **(files or {})}
    report = _fields(files["report"])
    validation = _read_csv(files["validation"])
    objectives = _read_csv(files["objectives"])
    audit_rows = _read_csv(files.get("audit_csv", ""))
    matched_rows = _read_csv(files["matched"])
    comparison_rows = _read_csv(files["comparison"])
    astra = _fields(files["astra"])

    validation_nonempty = len(validation) > 0
    objectives_ok, objective_mismatches = _objectives_reproducible(validation, objectives)
    best_ok = _report_best_fixed_consistent(report, validation, objectives)
    invalid_ok = _report_invalid_rate_consistent(report, validation, objectives)
    no_v9, v9_offenders = _no_stale_v9_paths(files)
    matched_exists = bool(matched_rows) and bool(_fields(files["matched_summary"])) and any(
        r.get("baseline_policy") == "best_fixed" for r in matched_rows
    )
    astra_sim_false = str(astra.get("ASTRA_SIM_RUN_COMPLETED", "false")).lower() == "false"
    astra_uses_schedule = str(astra.get("ASTRA_EXPORT_USES_REAL_SCHEDULE", "false")).lower() == "true"
    invalid_rows = _audit_invalid_rows(audit_rows)
    artifact_core_ok = validation_nonempty and objectives_ok and best_ok and invalid_ok and no_v9
    ready_ok = _ready_gate_trustworthy(
        report,
        artifact_core_ok=artifact_core_ok,
        matched_exists=matched_exists,
        astra_ok=astra_sim_false and astra_uses_schedule,
        invalid_rows=invalid_rows,
    )
    comparison_exists = bool(comparison_rows)
    status = "PASS" if artifact_core_ok and matched_exists and ready_ok and astra_sim_false else "FAIL"
    if status == "PASS" and (not comparison_exists or invalid_rows not in {0, -1}):
        status = "WARNING"

    fields: dict[str, Any] = {
        "ARTIFACT_SANITY": status,
        "VALIDATION_CSV_NONEMPTY": str(validation_nonempty).lower(),
        "OBJECTIVES_REPRODUCIBLE_FROM_VALIDATION": str(objectives_ok).lower(),
        "REPORT_BEST_FIXED_CONSISTENT": str(best_ok).lower(),
        "REPORT_INVALID_RATE_CONSISTENT": str(invalid_ok).lower(),
        "NO_STALE_PR4_V9_PATHS": str(no_v9).lower(),
        "MATCHED_COMPARISON_EXISTS": str(matched_exists).lower(),
        "READY_GATE_TRUSTWORTHY": str(ready_ok).lower(),
        "ASTRA_SIM_DECLARED_FALSE": str(astra_sim_false).lower(),
        "ASTRA_EXPORT_USES_REAL_SCHEDULE": str(astra_uses_schedule).lower(),
        "AGENTWEAVER_V11_COMPARISON_EXISTS": str(comparison_exists).lower(),
        "AUDIT_INVALID_ROWS": invalid_rows,
        "VALIDATION_ROWS": len(validation),
        "OBJECTIVE_ROWS": len(objectives),
        "OBJECTIVE_MISMATCHES": ";".join(objective_mismatches[:20]),
        "STALE_PR4_V9_PATHS": ";".join(v9_offenders[:20]),
    }
    lines = ["# PR4-v11 Artifact Sanity", ""]
    lines.extend(f"{key} = {value}" for key, value in fields.items())
    lines.extend(["", "## Inputs"])
    lines.extend(f"- {key}: {value}" for key, value in sorted(files.items()))
    p = Path(out_md)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=DEFAULT_FILES["report"])
    ap.add_argument("--validation", default=DEFAULT_FILES["validation"])
    ap.add_argument("--objectives", default=DEFAULT_FILES["objectives"])
    ap.add_argument("--matched", default=DEFAULT_FILES["matched"])
    ap.add_argument("--matched-summary", default=DEFAULT_FILES["matched_summary"])
    ap.add_argument("--comparison", default=DEFAULT_FILES["comparison"])
    ap.add_argument("--astra", default=DEFAULT_FILES["astra"])
    ap.add_argument("--out", default="data/results/pr4_v11_artifact_sanity.md")
    args = ap.parse_args()
    files = {
        "report": args.report,
        "validation": args.validation,
        "objectives": args.objectives,
        "matched": args.matched,
        "matched_summary": args.matched_summary,
        "comparison": args.comparison,
        "astra": args.astra,
    }
    print(json.dumps(run_sanity(files, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
