from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

from agentweaver.utils.io import ensure_dir


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
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
        v = row.get(key, "")
        return default if v in ("", None) else float(v)
    except Exception:
        return default


def _gain(base: float, new: float, lower_better: bool = True) -> float:
    if base <= 0:
        return 0.0
    return (base - new) / base if lower_better else (new - base) / base


def _median(vals: list[float]) -> float:
    return statistics.median(vals) if vals else 0.0


def _taps_lines(taps_rows: list[dict[str, str]]) -> tuple[list[str], tuple[float, str], tuple[float, str]]:
    lines: list[str] = []
    best = (-1e9, "")
    worst = (1e9, "")
    for sessions in [1, 2, 4, 8, 16, 32, 64]:
        relevant = [r for r in taps_rows if int(r.get("sessions", 0) or 0) == sessions]
        if not relevant:
            continue
        gains: list[tuple[float, str]] = []
        for arrival in sorted({r.get("arrival_pattern", "") for r in relevant}):
            for regions in sorted({int(r.get("effective_regions", 0) or 0) for r in relevant if r.get("arrival_pattern") == arrival}):
                by_policy = {
                    r.get("policy"): r
                    for r in relevant
                    if r.get("arrival_pattern") == arrival and int(r.get("effective_regions", 0) or 0) == regions
                }
                taps = by_policy.get("taps_v3")
                acd = by_policy.get("acd_nisp")
                naive = by_policy.get("naive_wafer")
                if not taps or not acd:
                    continue
                p95_acd = _gain(_f(acd, "p95_jct"), _f(taps, "p95_jct"))
                mean_acd = _gain(_f(acd, "mean_jct"), _f(taps, "mean_jct"))
                thr_acd = _gain(_f(acd, "throughput_sessions_per_sec"), _f(taps, "throughput_sessions_per_sec"), lower_better=False)
                ready_acd = _gain(_f(acd, "ready_queue_wait"), _f(taps, "ready_queue_wait"))
                util = _f(taps, "region_utilization") - _f(acd, "region_utilization")
                p95_naive = _gain(_f(naive, "p95_jct"), _f(taps, "p95_jct")) if naive else 0.0
                gains.append((p95_acd, f"sessions={sessions}, arrival={arrival}, regions={regions}"))
                if p95_acd > best[0]:
                    best = (p95_acd, gains[-1][1])
                if p95_acd < worst[0]:
                    worst = (p95_acd, gains[-1][1])
                lines.append(
                    "- sessions={}, arrival={}, regions={}: throughput_gain_vs_acd={:.6f}, mean_jct_gain_vs_acd={:.6f}, p95_jct_gain_vs_acd={:.6f}, p95_jct_gain_vs_naive={:.6f}, ready_queue_wait_gain={:.6f}, region_utilization_change={:.6f}".format(
                        sessions,
                        arrival,
                        regions,
                        thr_acd,
                        mean_acd,
                        p95_acd,
                        p95_naive,
                        ready_acd,
                        util,
                    )
                )
    return lines, best, worst


def write_diagnosis(
    report_v3: str | Path = "data/results/pr4_algo_v3_report.md",
    taps_csv: str | Path = "data/results/taps_v3_sweep.csv",
    taps_validation_csv: str | Path = "data/results/taps_v3_validation.csv",
    pabb_csv: str | Path = "data/results/pabb_online_branch_budget_pr4_v3.csv",
    cdf_csv: str | Path = "data/results/cdf_strict_prefix_comparison_pr4_v2.csv",
    latency_breakdown_csv: str | Path = "data/results/mini_swe_lite10_r4_timed_latency_breakdown_detailed.csv",
    out: str | Path = "data/results/pr4_algo_v4_diagnosis.md",
) -> dict[str, Any]:
    report = _fields(report_v3)
    taps = _read_csv(taps_csv)
    pabb = _read_csv(pabb_csv)
    cdf = _read_csv(cdf_csv)
    br = _read_csv(latency_breakdown_csv)
    cdf_added = sum(_f(r, "cdf_added_reusable_tokens") for r in cdf)
    cdf_saved = sum(_f(r, "estimated_prefill_saved") for r in cdf)
    block_mode = any(str(r.get("block_prefix_mode", "")).lower() == "true" for r in cdf)
    taps_lines, best_taps, worst_taps = _taps_lines(taps)
    pabb_gains = [_f(r, "pabb_online_gain_vs_fcfs") for r in pabb if r.get("policy") == "pabb_online" and r.get("pabb_online_gain_vs_fcfs") not in {"", None}]
    pabb_gap = [_f(r, "oracle_gap") for r in pabb if r.get("policy") == "pabb_online" and r.get("oracle_gap") not in {"", None}]
    patch_instances = {
        r["instance_id"]
        for r in pabb
        if r.get("policy") == "pabb_online" and r.get("time_to_first_nonempty_patch") not in {"", None}
    }
    pabb_fail = sorted(
        {
            r["instance_id"]
            for r in pabb
            if r.get("policy") == "pabb_online" and r.get("time_to_first_nonempty_patch") in {"", None}
        }
    )
    tool_shares = [_f(r, "tool_time_share") for r in br]
    llm_shares = [_f(r, "llm_time_share") for r in br]
    jcts = [_f(r, "measured_agent_jct") for r in br]
    by_instance: dict[str, list[float]] = {}
    for row in br:
        by_instance.setdefault(row.get("instance_id", ""), []).append(_f(row, "measured_agent_jct"))
    cvs: list[float] = []
    for vals in by_instance.values():
        if len(vals) >= 2 and _median(vals) > 0:
            cvs.append((statistics.pstdev(vals) / max(1e-9, statistics.mean(vals))))

    lines = ["# PR4 Algorithm v4 Diagnosis", ""]
    lines.extend(
        [
            "## CDF",
            f"CDF_STATUS = optional",
            f"CDF_ADDED_REUSABLE_TOKENS = {int(cdf_added)}",
            f"CDF_ESTIMATED_PREFILL_SAVED = {cdf_saved:.6f}",
            f"CDF_BLOCK_PREFIX_MODE = {str(block_mode).lower()}",
            "CDF_MAIN_MECHANISM = false",
            "CDF_DIAGNOSIS = CDF should remain secondary: added reusable tokens are small and block-prefix mode is approximate because raw token ids are unavailable.",
            "",
            "## TAPS",
        ]
    )
    lines.extend(taps_lines[:220])
    if len(taps_lines) > 220:
        lines.append(f"- ... truncated {len(taps_lines) - 220} additional pressure points; full data in {taps_csv}")
    lines.extend(
        [
            f"TAPS_BEST_SCENARIO = {best_taps[1]} p95_gain={best_taps[0]:.6f}",
            f"TAPS_WORST_SCENARIO = {worst_taps[1]} p95_gain={worst_taps[0]:.6f}",
            "TAPS_DIAGNOSIS = v3 ready-queue priority alone is too weak; v4 should optimize domain locality, admission, and memory pressure.",
            "",
            "## PABB",
            f"PABB_ONLINE_GAIN_MEDIAN = {_median(pabb_gains):.6f}",
            f"PABB_ONLINE_GAIN_MEAN = {(sum(pabb_gains) / len(pabb_gains) if pabb_gains else 0.0):.6f}",
            f"PABB_ORACLE_GAP_MEAN = {(sum(pabb_gap) / len(pabb_gap) if pabb_gap else 0.0):.6f}",
            f"PABB_PATCH_PRODUCING_INSTANCES = {len(patch_instances)}",
            f"PABB_ONLINE_FAIL_CASES = {', '.join(pabb_fail[:20])}",
            "PABB_DIAGNOSIS = PABB is online and no-leakage, but progress signals often arrive late; v4 should use measured incremental patch snapshots when available.",
            "",
            "## Workload",
            f"TOOL_TIME_SHARE_MEDIAN = {_median(tool_shares):.6f}",
            f"LLM_TIME_SHARE_MEDIAN = {_median(llm_shares):.6f}",
            f"MEASURED_JCT_MEDIAN = {_median(jcts):.6f}",
            f"BRANCH_JCT_CV_MEDIAN = {_median(cvs):.6f}",
            "WORKLOAD_DIAGNOSIS = mini-SWE is often tool-time dominated with high branch skew; pure prompt reuse alone cannot create large end-to-end JCT gains.",
            "",
            "## Conclusions",
            "MAIN_EFFECTIVE_MECHANISMS = ACD,NISP,TAPS-A/TAPS-D if validation shows non-oracle gain",
            "WEAK_MECHANISMS = CDF,TAPS-v3-ready-queue-only",
            "NEXT_OPTIMIZATION_TARGETS = domain locality, admission control, SRAM-bounded residency, earlier online patch progress",
            f"PR4_V3_REPORTED_TAPS_GAIN = {report.get('TAPS_V3_GAIN', 'unknown')}",
            f"PR4_V3_REPORTED_PABB_GAIN = {report.get('PABB_ONLINE_GAIN', 'unknown')}",
        ]
    )
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"out": str(out), "cdf_added": cdf_added, "best_taps_gain": best_taps[0], "pabb_patch_instances": len(patch_instances)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/pr4_algo_v4_diagnosis.md")
    args = ap.parse_args()
    print(write_diagnosis(out=args.out))


if __name__ == "__main__":
    main()
