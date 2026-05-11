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
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
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


def _gain(base: float, new: float, lower_better: bool = True) -> float:
    if base <= 0:
        return 0.0
    return (base - new) / base if lower_better else (new - base) / base


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _domain_summary(rows: list[dict[str, str]]) -> tuple[list[str], dict[str, float]]:
    by_key: dict[tuple[str, int, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (row.get("arrival_pattern", ""), int(_f(row, "effective_regions")), int(_f(row, "sessions")))
        by_key.setdefault(key, {})[row.get("policy", "")] = row
    p95: list[float] = []
    mean: list[float] = []
    ready: list[float] = []
    remote: list[float] = []
    worsened: list[str] = []
    for key, group in sorted(by_key.items()):
        base = group.get("acd_nisp")
        dom = group.get("taps_domain")
        if not base or not dom:
            continue
        p95_gain = _gain(_f(base, "p95_jct"), _f(dom, "p95_jct"))
        p95.append(p95_gain)
        mean.append(_gain(_f(base, "mean_jct"), _f(dom, "mean_jct")))
        ready.append(_gain(_f(base, "ready_queue_wait"), _f(dom, "ready_queue_wait")))
        remote.append(_gain(_f(base, "remote_kv_bytes"), _f(dom, "remote_kv_bytes")))
        if p95_gain < 0:
            worsened.append(f"arrival={key[0]}, regions={key[1]}, sessions={key[2]}, p95_gain={p95_gain:.6f}")
    lines = [
        "## Domain Scheduler",
        f"MEAN_JCT_GAIN_MEDIAN_VS_ACD_NISP = {_median(mean):.6f}",
        f"P95_JCT_GAIN_MEDIAN_VS_ACD_NISP = {_median(p95):.6f}",
        f"READY_QUEUE_WAIT_GAIN_MEDIAN_VS_ACD_NISP = {_median(ready):.6f}",
        f"REMOTE_KV_REDUCTION_MEDIAN_VS_ACD_NISP = {_median(remote):.6f}",
        f"P95_WORSENED_CONFIGS = {len(worsened)}",
    ]
    lines.extend(f"- {x}" for x in worsened[:30])
    return lines, {
        "domain_p95_median": _median(p95),
        "domain_p95_best": max(p95) if p95 else 0.0,
        "domain_remote_median": _median(remote),
        "domain_worsened": float(len(worsened)),
    }


def _admission_summary(rows: list[dict[str, str]]) -> tuple[list[str], dict[str, float]]:
    by_key: dict[tuple[int, int, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (int(_f(row, "total_sessions")), int(_f(row, "active_session_limit")), int(_f(row, "effective_regions")))
        by_key.setdefault(key, {})[row.get("policy", "")] = row
    vs_static_p95: list[float] = []
    vs_static_thr: list[float] = []
    vs_reactive_p95: list[float] = []
    vs_reactive_thr: list[float] = []
    reactive_wins = 0
    for group in by_key.values():
        taps = group.get("taps_admission")
        static = group.get("static_admission")
        reactive = group.get("reactive_admission")
        if taps and static:
            vs_static_p95.append(_gain(_f(static, "p95_jct"), _f(taps, "p95_jct")))
            vs_static_thr.append(_gain(_f(static, "throughput"), _f(taps, "throughput"), lower_better=False))
        if taps and reactive:
            p95_gain = _gain(_f(reactive, "p95_jct"), _f(taps, "p95_jct"))
            thr_gain = _gain(_f(reactive, "throughput"), _f(taps, "throughput"), lower_better=False)
            vs_reactive_p95.append(p95_gain)
            vs_reactive_thr.append(thr_gain)
            if p95_gain < 0 and thr_gain < 0:
                reactive_wins += 1
    lines = [
        "## Admission",
        f"TAPS_ADMISSION_P95_GAIN_MEDIAN_VS_STATIC = {_median(vs_static_p95):.6f}",
        f"TAPS_ADMISSION_THROUGHPUT_GAIN_MEDIAN_VS_STATIC = {_median(vs_static_thr):.6f}",
        f"TAPS_ADMISSION_P95_GAIN_MEDIAN_VS_REACTIVE = {_median(vs_reactive_p95):.6f}",
        f"TAPS_ADMISSION_THROUGHPUT_GAIN_MEDIAN_VS_REACTIVE = {_median(vs_reactive_thr):.6f}",
        f"REACTIVE_BEATS_TAPS_ADMISSION_CONFIGS = {reactive_wins}",
        "ADMISSION_DIAGNOSIS = taps_admission is strong versus static_admission, but reactive_admission is the stronger baseline and must be retained.",
    ]
    return lines, {
        "admission_p95_vs_reactive_median": _median(vs_reactive_p95),
        "admission_thr_vs_reactive_median": _median(vs_reactive_thr),
        "reactive_wins": float(reactive_wins),
    }


def _memory_summary(rows: list[dict[str, str]]) -> tuple[list[str], dict[str, float]]:
    lines = ["## Memory"]
    for row in rows:
        lines.append(
            "- budget={}GB: hit_rate={:.6f}, eviction_count={}, jct_over_64gb={:.6f}".format(
                row.get("memory_budget_gb", ""),
                _f(row, "hit_rate"),
                row.get("eviction_count", ""),
                _f(row, "jct_over_64gb"),
            )
        )
    max_jct = max([abs(_f(r, "jct_over_64gb")) for r in rows] or [0.0])
    max_hit = max([_f(r, "hit_rate") for r in rows] or [0.0])
    lines.append(
        "MEMORY_DIAGNOSIS = memory budget evaluation is present, but low hit_rate means TAPS-M is not a main performance source."
    )
    return lines, {"memory_max_jct_change": max_jct, "memory_max_hit": max_hit}


def _pabb_summary(rows: list[dict[str, str]]) -> tuple[list[str], dict[str, float]]:
    snapshot_events = sum(int(_f(r, "snapshot_events_available")) for r in rows)
    snap_rows = [r for r in rows if r.get("policy") in {"pabb_snapshot_online", "pabb_snapshot_online_v5"}]
    gains = [_f(r, "snapshot_gain_vs_fcfs") for r in snap_rows if r.get("snapshot_gain_vs_fcfs") not in {"", None}]
    lines = [
        "## PABB",
        f"PATCH_SNAPSHOT_EVENTS_AVAILABLE = {snapshot_events}",
        f"PABB_SNAPSHOT_GAIN_MEAN_VS_FCFS = {_mean(gains):.6f}",
        "PABB_DIAGNOSIS = snapshot progress remains weak when trace events do not contain real git snapshot fields.",
    ]
    if snapshot_events == 0:
        lines.append("PABB_REQUIRED_FIX = rerun mini-SWE with AGENTWEAVER_CAPTURE_PATCH_SNAPSHOTS=1 or patched tool hooks that record git status/diff-stat after tool events.")
    return lines, {"snapshot_events": float(snapshot_events), "pabb_gain": _mean(gains)}


def write_diagnosis(
    report_v4: str | Path = "data/results/pr4_algo_v4_report.md",
    domain_csv: str | Path = "data/results/taps_domain_scheduler_pr4_v4.csv",
    admission_csv: str | Path = "data/results/taps_admission_pr4_v4.csv",
    memory_csv: str | Path = "data/results/taps_memory_budget_pr4_v4.csv",
    pabb_csv: str | Path = "data/results/pabb_snapshot_online_pr4_v4.csv",
    out: str | Path = "data/results/pr4_algo_v5_diagnosis.md",
) -> dict[str, float]:
    report = _fields(report_v4)
    domain_lines, domain_stats = _domain_summary(_read_csv(domain_csv))
    admission_lines, admission_stats = _admission_summary(_read_csv(admission_csv))
    memory_lines, memory_stats = _memory_summary(_read_csv(memory_csv))
    pabb_lines, pabb_stats = _pabb_summary(_read_csv(pabb_csv))
    stats = {**domain_stats, **admission_stats, **memory_stats, **pabb_stats}
    strong: list[str] = ["ACD", "NISP"]
    if admission_stats["admission_thr_vs_reactive_median"] > 0 or admission_stats["admission_p95_vs_reactive_median"] > 0:
        strong.append("TAPS-A under some pressure")
    weak = ["CDF", "TAPS-M", "PABB-S"]
    if domain_stats["domain_worsened"] > 0:
        weak.append("TAPS-D p95 stability")
    lines = [
        "# PR4 Algorithm v5 Diagnosis",
        "",
        f"PR4_V4_GATE = {report.get('PR4_ALGO_V4_GATE', 'unknown')}",
        f"PR4_V4_READY_FOR_SCALE = {report.get('READY_FOR_PR4_SCALE', 'unknown')}",
        "",
    ]
    for section in (domain_lines, admission_lines, memory_lines, pabb_lines):
        lines.extend(section)
        lines.append("")
    lines.extend(
        [
            "## Final Diagnosis",
            f"STRONG_COMPONENTS = {','.join(strong)}",
            f"WEAK_COMPONENTS = {','.join(weak)}",
            "REQUIRED_FIXES = unify admission/domain/memory/tail scheduling; compare against reactive_admission and acd_nisp; rerun patch snapshot instrumentation or mark PABB-S weak.",
        ]
    )
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/pr4_algo_v5_diagnosis.md")
    args = ap.parse_args()
    print(write_diagnosis(out=args.out))


if __name__ == "__main__":
    main()
