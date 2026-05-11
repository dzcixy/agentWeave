from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agentweaver.utils.io import ensure_dir, write_csv


REQUIRED_POSITIVE = ["throughput", "p95_jct", "mean_jct"]
REQUIRED_NONNEGATIVE = ["p99_jct", "region_utilization"]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def _has_nan_or_inf(row: dict[str, Any]) -> bool:
    for key, value in row.items():
        if key in {"config_id", "arrival_pattern", "policy"}:
            continue
        if value in ("", None):
            continue
        try:
            f = float(value)
        except Exception:
            continue
        if not math.isfinite(f):
            return True
    return False


def audit_row(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    total = _float(row, "total_sessions")
    completed = _float(row, "completed_sessions")
    starvation = _float(row, "starvation_count")

    for key in ["total_sessions", "completed_sessions", "starvation_count", *REQUIRED_POSITIVE, *REQUIRED_NONNEGATIVE]:
        if math.isnan(_float(row, key)):
            reasons.append("missing_metric")
            break

    if math.isfinite(total) and math.isfinite(completed) and int(round(completed)) != int(round(total)):
        reasons.append("incomplete_sessions")
    if math.isfinite(starvation) and starvation != 0:
        reasons.append("starvation")

    for key in REQUIRED_POSITIVE:
        value = _float(row, key)
        if not math.isfinite(value) or value <= 0:
            reasons.append("invalid_metric")
    for key in REQUIRED_NONNEGATIVE:
        value = _float(row, key)
        if not math.isfinite(value) or value < 0:
            reasons.append("invalid_metric")

    p50 = _float(row, "p50_jct")
    p95 = _float(row, "p95_jct")
    p99 = _float(row, "p99_jct")
    if math.isfinite(p50) and math.isfinite(p95) and p95 < p50:
        reasons.append("invalid_metric")
    if math.isfinite(p95) and math.isfinite(p99) and p99 < p95:
        reasons.append("invalid_metric")
    if _has_nan_or_inf(row):
        reasons.append("invalid_metric")
    return not reasons, sorted(set(reasons))


def audit_grid(
    grid_csv: str | Path = "data/results/aligned_policy_grid_pr4_v10.csv",
    audit_csv: str | Path = "data/results/aligned_policy_grid_audit_pr4_v10.csv",
    valid_csv: str | Path = "data/results/aligned_policy_grid_valid_pr4_v10.csv",
    report_md: str | Path = "data/results/aligned_policy_grid_audit_pr4_v10.md",
) -> dict[str, Any]:
    rows = _read_csv(grid_csv)
    audit_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    invalid_by_policy: Counter[str] = Counter()
    invalid_by_config: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    policy_by_config: dict[str, set[str]] = defaultdict(set)
    valid_policy_by_config: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        cid = str(row.get("config_id", ""))
        policy = str(row.get("policy", ""))
        policy_by_config[cid].add(policy)
        valid, reasons = audit_row(row)
        out = dict(row)
        out["validity"] = str(valid).lower()
        out["invalid_reason"] = ";".join(reasons)
        audit_rows.append(out)
        if valid:
            valid_rows.append(dict(row))
            valid_policy_by_config[cid].add(policy)
        else:
            invalid_by_policy[policy] += 1
            invalid_by_config[cid] += 1
            for reason in reasons:
                reason_counts[reason] += 1

    all_policies = sorted({str(r.get("policy", "")) for r in rows if r.get("policy")})
    configs_all_valid = sum(1 for cid, policies in policy_by_config.items() if valid_policy_by_config.get(cid, set()) == policies and policies)
    configs_any_valid = sum(1 for cid in policy_by_config if valid_policy_by_config.get(cid))
    configs_all_policies_valid = sum(1 for cid in policy_by_config if set(all_policies).issubset(valid_policy_by_config.get(cid, set())))

    write_csv(audit_csv, audit_rows)
    write_csv(valid_csv, valid_rows)

    lines = [
        "# Aligned Policy Grid Audit PR4-v10",
        "",
        f"INPUT_GRID = {grid_csv}",
        f"TOTAL_ROWS = {len(rows)}",
        f"VALID_ROWS = {len(valid_rows)}",
        f"INVALID_ROWS = {len(rows) - len(valid_rows)}",
        f"VALID_CONFIGS_ALL_RECORDED_POLICIES = {configs_all_valid}",
        f"VALID_CONFIGS_ALL_POLICIES = {configs_all_policies_valid}",
        f"CONFIGS_WITH_AT_LEAST_ONE_VALID_POLICY = {configs_any_valid}",
        "",
        "## Invalid Rows By Policy",
    ]
    if invalid_by_policy:
        lines.extend(f"- {policy}: {count}" for policy, count in sorted(invalid_by_policy.items()))
    else:
        lines.append("- none")
    lines.extend(["", "## Invalid Reasons", ""])
    if reason_counts:
        lines.extend(f"- {reason}: {count}" for reason, count in sorted(reason_counts.items()))
    else:
        lines.append("- none")
    frequent_invalid = [policy for policy, count in invalid_by_policy.items() if count >= max(1, len(rows) // max(1, len(all_policies)) // 10)]
    lines.extend(
        [
            "",
            "## Policy Validity Rule",
            "Rows are valid only when all sessions complete, starvation is zero, core latency/throughput metrics are positive, ordering constraints hold, and no NaN/inf appears.",
            "Invalid rows are kept in the audit CSV but excluded from best-fixed, oracle-envelope, and default cost-model training.",
            "",
            "## Policies Frequently Invalid",
            ", ".join(sorted(frequent_invalid)) if frequent_invalid else "none",
        ]
    )
    p = Path(report_md)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "total_rows": len(rows),
        "valid_rows": len(valid_rows),
        "invalid_rows": len(rows) - len(valid_rows),
        "valid_configs_all_policies": configs_all_policies_valid,
        "configs_with_at_least_one_valid_policy": configs_any_valid,
        "invalid_by_policy": dict(invalid_by_policy),
        "invalid_reasons": dict(reason_counts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="data/results/aligned_policy_grid_pr4_v10.csv")
    ap.add_argument("--audit-out", default="data/results/aligned_policy_grid_audit_pr4_v10.csv")
    ap.add_argument("--valid-out", default="data/results/aligned_policy_grid_valid_pr4_v10.csv")
    ap.add_argument("--report", default="data/results/aligned_policy_grid_audit_pr4_v10.md")
    args = ap.parse_args()
    summary = audit_grid(args.grid, args.audit_out, args.valid_out, args.report)
    print(summary)


if __name__ == "__main__":
    main()
