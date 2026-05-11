from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from agentweaver.utils.io import ensure_dir


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _field_report(path: str | Path) -> dict[str, str]:
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
        value = row.get(key, "")
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def _gain(base: float, new: float, lower_better: bool = True) -> float:
    if base == 0:
        return 0.0
    return (base - new) / base if lower_better else (new - base) / base


def _detect_pabb_leakage_source(path: str | Path = "agentweaver/simulator/progress_aware_branch_budgeting.py") -> tuple[bool, list[str]]:
    text = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
    leakage_fields = [
        "patch_nonempty",
        "git_diff_bytes",
        "duplicate_patch_hash",
        "no_file_modification",
        "llm_tokens_used",
        "official_verifier_result",
    ]
    offending: list[str] = []
    for field in leakage_fields:
        if re.search(rf"\b{re.escape(field)}\b", text):
            offending.append(field)
    has_leak = "def _utility(sig: BranchSignals" in text and bool(offending)
    return has_leak, offending


def write_diagnosis(
    v2_report: str | Path = "data/results/pr4_algo_v2_report.md",
    taps_csv: str | Path = "data/results/multisession_taps_predictive_pr4_v2.csv",
    pabb_csv: str | Path = "data/results/pabb_branch_budget_pr4_algo.csv",
    cdf_csv: str | Path = "data/results/cdf_strict_prefix_comparison_pr4_v2.csv",
    out: str | Path = "data/results/pr4_algo_v3_diagnosis.md",
) -> dict[str, Any]:
    fields = _field_report(v2_report)
    taps = _read_csv(taps_csv)
    pabb = _read_csv(pabb_csv)
    cdf = _read_csv(cdf_csv)
    cdf_added = sum(_f(r, "cdf_added_reusable_tokens") for r in cdf)
    cdf_saved = sum(_f(r, "estimated_prefill_saved") for r in cdf)
    block_mode = any(str(r.get("block_prefix_mode", "")).lower() == "true" for r in cdf)
    leakage, leakage_fields = _detect_pabb_leakage_source()

    lines = ["# PR4 Algorithm v3 Diagnosis", ""]
    lines.extend(
        [
            "## CDF",
            f"added_reusable_tokens = {int(cdf_added)}",
            f"model_side_speedup = {fields.get('CDF_MODEL_SIDE_SPEEDUP', 'unknown')}",
            f"estimated_prefill_saved = {cdf_saved:.6f}",
            f"block_prefix_mode = {str(block_mode).lower()}",
            "diagnosis = CDF is implemented with strict/block-prefix accounting, but the observed mini-SWE gain is weak because these traces already render most stable task/tool/repo context near the prompt prefix and the remaining canonicalizable context is small relative to total tool time.",
            "paper_ready = secondary_only",
            "",
            "## TAPS",
        ]
    )
    for sessions in sorted({int(r.get("sessions", 0) or 0) for r in taps}):
        by_policy = {r.get("policy"): r for r in taps if int(r.get("sessions", 0) or 0) == sessions}
        base = by_policy.get("acd_nisp") or by_policy.get("naive_wafer")
        pred = by_policy.get("taps_predictive") or by_policy.get("taps_predictive_v2") or {}
        lines.append(
            "- sessions={}: throughput_gain={:.6f}, mean_jct_gain={:.6f}, p95_jct_gain={:.6f}, ready_queue_wait_gain={:.6f}, region_utilization_change={:.6f}".format(
                sessions,
                _gain(_f(base, "throughput_sessions_per_sec"), _f(pred, "throughput_sessions_per_sec"), lower_better=False),
                _gain(_f(base, "mean_jct"), _f(pred, "mean_jct")),
                _gain(_f(base, "p95_jct"), _f(pred, "p95_jct")),
                _gain(_f(base, "ready_queue_wait"), _f(pred, "ready_queue_wait")),
                _f(pred, "region_utilization") - _f(base, "region_utilization"),
            )
        )
    lines.extend(
        [
            "diagnosis = TAPS is strongest under high session pressure where tool stalls create ready-queue contention. It is weak at low session counts because the trace is tool-time dominated and there is little hidden work to schedule.",
            "paper_ready = needs_v3_predictive_stress_test",
            "",
            "## PABB",
            f"legacy_rows = {len(pabb)}",
            f"CURRENT_PABB_HAS_FUTURE_LEAKAGE = {str(leakage).lower()}",
            "leakage_fields = " + ", ".join(leakage_fields),
            "diagnosis = The legacy pabb_budget path ranks branches using full-branch summaries before those branches have executed. It is invalid as an online scheduler and must be replaced by event-level replay.",
            "paper_ready = false_until_v3_online_tests_pass",
        ]
    )
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"cdf_added": cdf_added, "block_prefix_mode": block_mode, "pabb_leakage": leakage, "out": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/pr4_algo_v3_diagnosis.md")
    args = ap.parse_args()
    result = write_diagnosis(out=args.out)
    print(result)


if __name__ == "__main__":
    main()
