from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agentweaver.utils.io import ensure_dir


DEFAULT_FILES = {
    "report": "data/results/pr4_algo_v9_report.md",
    "methodology": "data/results/pr4_v9_evaluation_methodology.md",
    "grid": "data/results/aligned_policy_grid_stratified_pr4_v9.csv",
    "valid_grid": "data/results/aligned_policy_grid_stratified_valid_pr4_v9.csv",
    "validation": "data/results/taps_compiler_v2_validation_pr4_v9.csv",
    "objectives": "data/results/taps_compiler_v2_objectives_pr4_v9.csv",
    "audit": "data/results/aligned_policy_grid_stratified_audit_pr4_v9.csv",
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
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        if re.fullmatch(r"[A-Z0-9_]+", key.strip()):
            out[key.strip()] = value.strip()
    return out


def _float(value: Any, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _validation_invalid_rate(rows: list[dict[str, str]], objective: str = "balanced") -> float:
    vals = [r for r in rows if r.get("objective") == objective]
    if not vals:
        return math.nan
    return sum(str(r.get("invalid_selection", "")).lower() == "true" for r in vals) / len(vals)


def _audit_counts(rows: list[dict[str, str]]) -> dict[str, Any]:
    valid = 0
    invalid = 0
    by_policy: Counter[str] = Counter()
    for row in rows:
        if str(row.get("validity", "")).lower() == "true":
            valid += 1
        else:
            invalid += 1
            by_policy[str(row.get("policy", ""))] += 1
    return {"valid_rows": valid, "invalid_rows": invalid, "invalid_by_policy": dict(by_policy)}


def _objective_best_fixed(rows: list[dict[str, str]], split: str = "random", objective: str = "balanced") -> str:
    for row in rows:
        if row.get("split_type") == split and row.get("objective") == objective:
            return str(row.get("best_fixed_policy", ""))
    return ""


def _objective_invalid_rate(rows: list[dict[str, str]], split: str = "random", objective: str = "balanced") -> float:
    for row in rows:
        if row.get("split_type") == split and row.get("objective") == objective:
            return _float(row.get("invalid_selection_rate"))
    return math.nan


def _default_paths_use_current_pr() -> bool:
    checks = {
        "agentweaver/simulator/aligned_policy_sweep.py": [
            "aligned_policy_grid_pr4_v10.csv",
            "aligned_policy_grid_summary_pr4_v10.md",
        ],
        "agentweaver/analysis/policy_grid_audit.py": [
            "aligned_policy_grid_pr4_v10.csv",
            "aligned_policy_grid_audit_pr4_v10.csv",
            "aligned_policy_grid_valid_pr4_v10.csv",
        ],
        "agentweaver/simulator/workload_feature_extractor.py": [
            "aligned_policy_grid_valid_pr4_v10.csv",
            "workload_features_pr4_v10.csv",
        ],
    }
    for path, needles in checks.items():
        text = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
        if not all(needle in text for needle in needles):
            return False
    return True


def run_sanity(
    files: dict[str, str] | None = None,
    out_md: str | Path = "data/results/pr4_v10_artifact_sanity.md",
) -> dict[str, Any]:
    files = {**DEFAULT_FILES, **(files or {})}
    report = _fields(files["report"])
    methodology = _fields(files["methodology"])
    validation = _read_csv(files["validation"])
    objectives = _read_csv(files["objectives"])
    audit = _read_csv(files["audit"])
    grid = _read_csv(files["grid"])
    valid_grid = _read_csv(files["valid_grid"])

    validation_nonempty = len(validation) > 0
    objective_best = _objective_best_fixed(objectives)
    report_best = report.get("BEST_FIXED_POLICY", "")
    methodology_best = methodology.get("BEST_FIXED_POLICY", "")
    best_consistent = bool(report_best and objective_best and methodology_best and report_best == objective_best == methodology_best)

    report_invalid = _float(report.get("INVALID_SELECTION_RATE"))
    methodology_invalid = _float(methodology.get("INVALID_SELECTION_RATE"))
    objective_invalid = _objective_invalid_rate(objectives)
    validation_invalid = _validation_invalid_rate(validation)
    invalid_values = [report_invalid, methodology_invalid, objective_invalid, validation_invalid]
    invalid_consistent = all(math.isfinite(v) for v in invalid_values) and (max(invalid_values) - min(invalid_values) <= 1e-6)

    audit_counts = _audit_counts(audit)
    counts_consistent = (
        int(_float(report.get("VALID_ROWS"), -1)) == audit_counts["valid_rows"]
        and int(_float(report.get("INVALID_ROWS"), -1)) == audit_counts["invalid_rows"]
        and len(grid) == len(audit)
        and len(valid_grid) == audit_counts["valid_rows"]
    )
    default_paths_ok = _default_paths_use_current_pr()
    report_ready = str(report.get("READY_FOR_PR4_SCALE", "false")).lower() == "true"
    real_gate = validation_nonempty and best_consistent and invalid_consistent and counts_consistent and default_paths_ok
    ready_gate_trustworthy = (not report_ready) or real_gate
    status = "PASS" if real_gate and ready_gate_trustworthy else "FAIL"
    if status == "PASS" and report_ready and _float(report.get("RANDOM_SPLIT_P95_GAIN_OVER_BEST_FIXED"), 0.0) < 0.03:
        status = "WARNING"
        ready_gate_trustworthy = False
    fields: dict[str, Any] = {
        "ARTIFACT_SANITY": status,
        "VALIDATION_CSV_NONEMPTY": str(validation_nonempty).lower(),
        "REPORT_BEST_FIXED_CONSISTENT": str(best_consistent).lower(),
        "REPORT_INVALID_RATE_CONSISTENT": str(invalid_consistent).lower(),
        "DEFAULT_PATHS_USE_CURRENT_PR": str(default_paths_ok).lower(),
        "READY_GATE_TRUSTWORTHY": str(ready_gate_trustworthy).lower(),
        "VALID_ROWS_MATCH_AUDIT": str(counts_consistent).lower(),
        "REPORT_BEST_FIXED_POLICY": report_best,
        "OBJECTIVE_BEST_FIXED_POLICY": objective_best,
        "METHODOLOGY_BEST_FIXED_POLICY": methodology_best,
        "REPORT_INVALID_SELECTION_RATE": f"{report_invalid:.6f}" if math.isfinite(report_invalid) else "nan",
        "OBJECTIVE_INVALID_SELECTION_RATE": f"{objective_invalid:.6f}" if math.isfinite(objective_invalid) else "nan",
        "METHODOLOGY_INVALID_SELECTION_RATE": f"{methodology_invalid:.6f}" if math.isfinite(methodology_invalid) else "nan",
        "VALIDATION_INVALID_SELECTION_RATE": f"{validation_invalid:.6f}" if math.isfinite(validation_invalid) else "nan",
        "AUDIT_COUNTS": json.dumps(audit_counts, sort_keys=True),
    }
    lines = ["# PR4-v10 Artifact Sanity", ""]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    lines.extend(
        [
            "",
            "## Inputs",
            *[f"- {k}: {v}" for k, v in sorted(files.items())],
        ]
    )
    p = Path(out_md)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/pr4_v10_artifact_sanity.md")
    args = ap.parse_args()
    print(json.dumps(run_sanity(out_md=args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
