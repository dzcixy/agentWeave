from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agentweaver.utils.io import ensure_dir, write_csv


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


def _safe_gain(base: float, new: float, lower_better: bool = True) -> float:
    if base <= 0:
        return 0.0
    return (base - new) / base if lower_better else (new - base) / base


def _regime(row: dict[str, Any]) -> str:
    if _f(row, "memory_pressure") >= 0.8:
        return "MEMORY_PRESSURE"
    if _f(row, "blocked_session_fraction") >= 0.30 and _f(row, "region_utilization") <= 0.75:
        return "ADMISSION_STARVED"
    if _f(row, "ready_queue_pressure") >= 15 or _f(row, "p95_jct") >= 120:
        return "TAIL_RISK"
    if _f(row, "domain_cache_hit_rate") >= 0.85 and _f(row, "remote_kv_pressure") >= 0.15:
        return "DOMAIN_HOT"
    return "BALANCED"


def _v2_predictive_map(rows: list[dict[str, str]]) -> dict[tuple[int, str], dict[str, str]]:
    out: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        if row.get("policy") in {"taps_predictive_v2", "taps_predictive"}:
            out[(int(_f(row, "sessions")), row.get("arrival_pattern", ""))] = row
    return out


def write_regret_analysis(
    taps_v5_csv: str | Path = "data/results/taps_unified_pr4_v5.csv",
    domain_v4_csv: str | Path = "data/results/taps_domain_scheduler_pr4_v4.csv",
    admission_v4_csv: str | Path = "data/results/taps_admission_pr4_v4.csv",
    predictive_v2_csv: str | Path = "data/results/multisession_taps_predictive_pr4_v2.csv",
    out_csv: str | Path = "data/results/taps_policy_regret_pr4_v6.csv",
    out_md: str | Path = "data/results/taps_policy_regret_pr4_v6.md",
) -> list[dict[str, Any]]:
    v5 = _read_csv(taps_v5_csv)
    # The v4 files are read intentionally: the report records their existence and uses
    # v5-aligned replays where available to avoid comparing different simulator axes.
    domain_v4 = _read_csv(domain_v4_csv)
    admission_v4 = _read_csv(admission_v4_csv)
    pred_v2 = _v2_predictive_map(_read_csv(predictive_v2_csv))
    by_key: dict[tuple[int, int, int, str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in v5:
        key = (
            int(_f(row, "total_sessions")),
            int(_f(row, "active_session_limit")),
            int(_f(row, "effective_regions")),
            row.get("arrival_pattern", ""),
            int(_f(row, "memory_budget_gb")),
        )
        policy = "taps_unified_v5" if row.get("policy") == "taps_unified" else row.get("policy", "")
        by_key[key][policy] = row
    out: list[dict[str, Any]] = []
    for key, group in sorted(by_key.items()):
        total, limit, regions, arrival, memory = key
        pred = pred_v2.get((total, arrival))
        if pred:
            group["taps_predictive_v2"] = pred
        policies = {k: v for k, v in group.items() if k in {
            "reactive_admission",
            "acd_nisp",
            "taps_domain_v4",
            "taps_admission_v4",
            "taps_predictive_v2",
            "taps_unified_v5",
        }}
        if "taps_unified_v5" not in policies:
            continue
        best_p95_policy, best_p95 = min(policies.items(), key=lambda kv: _f(kv[1], "p95_jct"))
        throughput_key = "throughput_sessions_per_sec" if "throughput_sessions_per_sec" in next(iter(policies.values())) else "throughput"
        best_thr_policy, best_thr = max(
            policies.items(), key=lambda kv: _f(kv[1], "throughput" if "throughput" in kv[1] else "throughput_sessions_per_sec")
        )
        best_wait_policy, best_wait = min(policies.items(), key=lambda kv: _f(kv[1], "ready_queue_wait", float("inf")))
        taps = policies["taps_unified_v5"]
        base = taps
        cache_tokens = _f(base, "cache_hit_tokens") + _f(base, "recompute_tokens")
        row = {
            "total_sessions": total,
            "active_session_limit": limit,
            "effective_regions": regions,
            "arrival_pattern": arrival,
            "memory_budget": memory,
            "baseline_policies": ",".join(sorted(policies)),
            "best_policy_by_p95": best_p95_policy,
            "best_policy_by_throughput": best_thr_policy,
            "best_policy_by_ready_wait": best_wait_policy,
            "taps_unified_regret_p95": _safe_gain(_f(taps, "p95_jct"), _f(best_p95, "p95_jct")),
            "taps_unified_regret_throughput": _safe_gain(
                _f(best_thr, "throughput" if "throughput" in best_thr else "throughput_sessions_per_sec"),
                _f(taps, "throughput"),
                lower_better=False,
            ),
            "taps_unified_regret_ready_wait": _safe_gain(_f(taps, "ready_queue_wait"), _f(best_wait, "ready_queue_wait")),
            "ready_queue_pressure": _f(base, "ready_queue_wait") / max(1, total),
            "blocked_session_fraction": _f(base, "blocked_session_fraction"),
            "region_utilization": _f(base, "region_utilization"),
            "domain_cache_hit_rate": _f(base, "domain_cache_hit_rate"),
            "remote_kv_pressure": _f(base, "remote_kv_bytes") / max(1.0, cache_tokens * 1024),
            "memory_pressure": _f(base, "memory_occupancy") / max(1.0, memory * 1024**3),
            "session_pressure": total / max(1, limit),
            "region_pressure": limit / max(1, regions),
            "p95_jct": _f(base, "p95_jct"),
        }
        row["regime"] = _regime(row)
        out.append(row)
    write_csv(out_csv, out)
    best_by_regime: dict[str, Counter[str]] = defaultdict(Counter)
    for row in out:
        best_by_regime[row["regime"]][row["best_policy_by_p95"]] += 1
    lines = [
        "# TAPS Policy Regret PR4-v6",
        "",
        f"CONFIGS_ANALYZED = {len(out)}",
        f"DOMAIN_V4_ROWS = {len(domain_v4)}",
        f"ADMISSION_V4_ROWS = {len(admission_v4)}",
        f"REGIMES_IDENTIFIED = {','.join(sorted({r['regime'] for r in out}))}",
        "",
        "## Best Policy By Regime",
    ]
    for regime, counts in sorted(best_by_regime.items()):
        lines.append(f"- {regime}: " + ", ".join(f"{k}={v}" for k, v in counts.most_common()))
    regret_sources = Counter()
    for row in out:
        if row["best_policy_by_p95"] != "taps_unified_v5":
            regret_sources[row["best_policy_by_p95"]] += 1
    lines.extend(
        [
            "",
            "## TAPS-U-v5 Regret Sources",
            ", ".join(f"{k}={v}" for k, v in regret_sources.most_common()) or "none",
            "",
            "## Interpretation",
            "- Regime labels are derived from current-state pressure features, not future JCT.",
            "- v4 CSVs are used for diagnosis context; aligned v5 replay rows are preferred for per-config regret to avoid mixing simulator axes.",
            "- If one policy dominates all regimes, the report says so rather than forcing artificial regime separation.",
        ]
    )
    p = Path(out_md)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", default="data/results/taps_policy_regret_pr4_v6.csv")
    ap.add_argument("--out-md", default="data/results/taps_policy_regret_pr4_v6.md")
    args = ap.parse_args()
    rows = write_regret_analysis(out_csv=args.out_csv, out_md=args.out_md)
    print({"rows": len(rows), "out": args.out_csv})


if __name__ == "__main__":
    main()
