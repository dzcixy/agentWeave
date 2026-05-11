from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agentweaver.simulator.aligned_policy_sweep import ALIGNED_POLICIES
from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.io import ensure_dir


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
        value = row.get(key)
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def _objective(row: dict[str, Any], med: dict[str, float]) -> float:
    p95 = _f(row, "p95_jct") / max(1e-9, med.get("p95_jct", 1.0))
    mean = _f(row, "mean_jct") / max(1e-9, med.get("mean_jct", 1.0))
    throughput = _f(row, "throughput") / max(1e-9, med.get("throughput", 1.0))
    return p95 + 0.2 * mean - 0.2 * throughput


def _median(vals: list[float]) -> float:
    vals = sorted(v for v in vals if math.isfinite(v) and v > 0)
    if not vals:
        return 1.0
    return vals[len(vals) // 2]


def _split(configs: list[str]) -> tuple[set[str], set[str]]:
    train, val = set(), set()
    for cid in sorted(configs):
        (val if int(stable_hash(cid), 16) % 5 == 0 else train).add(cid)
    if not train or not val:
        for i, cid in enumerate(sorted(configs)):
            (val if i % 5 == 0 else train).add(cid)
    return train, val


def write_methodology(
    valid_grid: str | Path = "data/results/aligned_policy_grid_valid_pr4_v9.csv",
    audit_report: str | Path = "data/results/aligned_policy_grid_audit_pr4_v9.md",
    out: str | Path = "data/results/pr4_v9_evaluation_methodology.md",
) -> dict[str, Any]:
    rows = _read_csv(valid_grid)
    by_config: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_config[str(row.get("config_id", ""))][str(row.get("policy", ""))] = row
    configs = sorted(by_config)
    train, val = _split(configs)
    train_rows = [r for cid in train for r in by_config[cid].values()]
    med = {
        "p95_jct": _median([_f(r, "p95_jct") for r in train_rows]),
        "mean_jct": _median([_f(r, "mean_jct") for r in train_rows]),
        "throughput": _median([_f(r, "throughput") for r in train_rows]),
    }
    objective_by_policy: dict[str, list[float]] = defaultdict(list)
    for cid in train:
        for policy, row in by_config[cid].items():
            objective_by_policy[policy].append(_objective(row, med))
    avg_obj = {
        policy: sum(vals) / len(vals)
        for policy, vals in objective_by_policy.items()
        if vals
    }
    best_fixed = min(avg_obj, key=avg_obj.get) if avg_obj else ""

    gains_best_fixed: list[float] = []
    gains_reactive: list[float] = []
    gains_acd: list[float] = []
    regrets: list[float] = []
    worst_regret = 0.0
    invalid_selection_rate = 0.0
    for cid in val:
        group = by_config[cid]
        best_row = group.get(best_fixed)
        oracle_row = min(group.values(), key=lambda r: _f(r, "p95_jct", float("inf"))) if group else None
        reactive = group.get("reactive_admission")
        acd = group.get("acd_nisp")
        if best_row and oracle_row:
            gain = (_f(best_row, "p95_jct") - _f(best_row, "p95_jct")) / max(1e-9, _f(best_row, "p95_jct"))
            gains_best_fixed.append(gain)
            regret = (_f(best_row, "p95_jct") - _f(oracle_row, "p95_jct")) / max(1e-9, _f(oracle_row, "p95_jct"))
            regrets.append(regret)
            worst_regret = max(worst_regret, regret)
        if best_row and reactive:
            gains_reactive.append((_f(reactive, "p95_jct") - _f(best_row, "p95_jct")) / max(1e-9, _f(reactive, "p95_jct")))
        if best_row and acd:
            gains_acd.append((_f(acd, "p95_jct") - _f(best_row, "p95_jct")) / max(1e-9, _f(acd, "p95_jct")))
        if best_fixed and best_fixed not in group:
            invalid_selection_rate += 1.0
    invalid_selection_rate = invalid_selection_rate / max(1, len(val))

    policy_coverage = Counter()
    for group in by_config.values():
        for policy in group:
            policy_coverage[policy] += 1

    def avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    fields = {
        "VALID_GRID": str(valid_grid),
        "TRAIN_CONFIGS": len(train),
        "VALIDATION_CONFIGS": len(val),
        "FIXED_DEPLOYABLE_BASELINES": ",".join(ALIGNED_POLICIES),
        "BEST_FIXED_POLICY": best_fixed,
        "BEST_FIXED_SELECTED_ON_TRAIN_ONLY": "true",
        "ORACLE_ENVELOPE_NOT_BASELINE": "true",
        "INVALID_ROWS_EXCLUDED": "true",
        "BEST_FIXED_GAIN_OVER_REACTIVE_P95": f"{avg(gains_reactive):.6f}",
        "BEST_FIXED_GAIN_OVER_ACD_NISP_P95": f"{avg(gains_acd):.6f}",
        "BEST_FIXED_REGRET_TO_ORACLE_P95": f"{avg(regrets):.6f}",
        "WORST_CASE_REGRET": f"{worst_regret:.6f}",
        "INVALID_SELECTION_RATE": f"{invalid_selection_rate:.6f}",
    }
    lines = [
        "# PR4-v9 Evaluation Methodology",
        "",
        "This methodology replaces the PR4-v8 wording that made the per-configuration strongest policy look like a deployable baseline.",
        "",
    ]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    lines.extend(
        [
            "",
            "## Baseline Classes",
            "1. Fixed deployable baselines: one policy is chosen before validation and applied unchanged.",
            "2. Best fixed policy: selected on the train split by normalized objective, then applied to validation configs.",
            "3. Oracle envelope: the best valid policy per validation config after outcomes are known. It is an upper bound only, not a deployable baseline.",
            "",
            "## Validity",
            f"Invalid/starved/incomplete rows are excluded from best-fixed selection, oracle-envelope computation, and default cost-model training. See {audit_report}.",
            "",
            "## Policy Coverage In Valid Grid",
        ]
    )
    lines.extend(f"- {policy}: {count}" for policy, count in sorted(policy_coverage.items()))
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-grid", default="data/results/aligned_policy_grid_valid_pr4_v9.csv")
    ap.add_argument("--out", default="data/results/pr4_v9_evaluation_methodology.md")
    args = ap.parse_args()
    print(write_methodology(args.valid_grid, out=args.out))


if __name__ == "__main__":
    main()
