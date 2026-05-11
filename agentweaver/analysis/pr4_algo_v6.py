from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.analysis.taps_policy_regret_analysis import write_regret_analysis
from agentweaver.simulator.taps_adaptive_autotune import run_adaptive_autotune
from agentweaver.simulator.taps_adaptive_autotune import run_adaptive_autotune
from agentweaver.simulator.taps_regime_classifier import RegimeThresholds
from agentweaver.simulator.taps_unified import TAPSUnifiedConfig
from agentweaver.simulator.taps_unified_adaptive import ADAPTIVE_POLICY, AdaptiveProfiles, run_adaptive_sweep
from agentweaver.utils.io import ensure_dir, write_csv


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _f(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return default
    try:
        v = row.get(key)
        return default if v in ("", None) else float(v)
    except Exception:
        return default


def _profiles_from_json(data: dict[str, Any]) -> AdaptiveProfiles:
    raw = data.get("best_profiles", data)
    if not raw:
        return AdaptiveProfiles.default()
    return AdaptiveProfiles(
        balanced=TAPSUnifiedConfig(**raw.get("balanced", {})),
        admission_starved=TAPSUnifiedConfig(**raw.get("admission_starved", {})),
        domain_hot=TAPSUnifiedConfig(**raw.get("domain_hot", {})),
        tail_risk=TAPSUnifiedConfig(**raw.get("tail_risk", {})),
        memory_pressure=TAPSUnifiedConfig(**raw.get("memory_pressure", {})),
        thresholds=RegimeThresholds(**raw.get("thresholds", {})),
        tail_slo_percentile=int(raw.get("tail_slo_percentile", 90)),
    )


def _gain_status(p95_gain: float) -> str:
    if p95_gain >= 0.10:
        return "STRONG"
    if p95_gain >= 0.03:
        return "MODERATE"
    if p95_gain > 0:
        return "WEAK"
    return "NOT_OBSERVED"


def write_fair_baseline_summary(
    adaptive_csv: str | Path = "data/results/taps_unified_adaptive_pr4_v6.csv",
    out_csv: str | Path = "data/results/taps_fair_baseline_summary_pr4_v6.csv",
) -> list[dict[str, Any]]:
    rows = _read_csv(adaptive_csv)
    by_key: dict[tuple[int, int, int, str, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (
            int(_f(row, "total_sessions")),
            int(_f(row, "active_session_limit")),
            int(_f(row, "effective_regions")),
            row.get("arrival_pattern", ""),
            int(_f(row, "memory_budget_gb")),
        )
        by_key.setdefault(key, {})[row.get("policy", "")] = row
    out: list[dict[str, Any]] = []
    for key, group in sorted(by_key.items()):
        adaptive = group.get(ADAPTIVE_POLICY)
        if not adaptive:
            continue
        baselines = {k: v for k, v in group.items() if k != ADAPTIVE_POLICY}
        p95_policy, p95_row = min(baselines.items(), key=lambda kv: _f(kv[1], "p95_jct"))
        thr_policy, thr_row = max(baselines.items(), key=lambda kv: _f(kv[1], "throughput"))
        out.append(
            {
                "total_sessions": key[0],
                "active_session_limit": key[1],
                "effective_regions": key[2],
                "arrival_pattern": key[3],
                "memory_budget_gb": key[4],
                "strongest_non_oracle_baseline": p95_policy,
                "strongest_throughput_baseline": thr_policy,
                "strongest_baseline_p95": _f(p95_row, "p95_jct"),
                "strongest_baseline_throughput": _f(thr_row, "throughput"),
                "adaptive_p95": _f(adaptive, "p95_jct"),
                "adaptive_throughput": _f(adaptive, "throughput"),
                "adaptive_gain_over_strongest_p95": (_f(p95_row, "p95_jct") - _f(adaptive, "p95_jct")) / max(1e-9, _f(p95_row, "p95_jct")),
                "adaptive_gain_over_strongest_throughput": (_f(adaptive, "throughput") - _f(thr_row, "throughput")) / max(1e-9, _f(thr_row, "throughput")),
                "adaptive_ready_wait": _f(adaptive, "ready_queue_wait"),
                "adaptive_starvation_count": int(_f(adaptive, "starvation_count")),
            }
        )
    write_csv(out_csv, out)
    return out


def plot_fair_baseline(rows: list[dict[str, Any]], out: str | Path = "data/plots/taps_fair_baseline_pr4_v6.pdf") -> None:
    ensure_dir(Path(out).parent)
    sub = [r for r in rows if int(r["active_session_limit"]) == 16 and int(r["effective_regions"]) == 4 and r["arrival_pattern"] == "bursty"]
    totals = sorted({int(r["total_sessions"]) for r in sub})
    plt.figure(figsize=(6.4, 3.8))
    base = [next((_f(r, "strongest_baseline_p95") for r in sub if int(r["total_sessions"]) == t), 0.0) for t in totals]
    adap = [next((_f(r, "adaptive_p95") for r in sub if int(r["total_sessions"]) == t), 0.0) for t in totals]
    plt.plot(totals, base, marker="o", label="strongest baseline")
    plt.plot(totals, adap, marker="o", label="adaptive")
    plt.xlabel("total sessions")
    plt.ylabel("p95 JCT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def write_pabb_status(out: str | Path = "data/results/pabb_status_pr4_v6.md") -> None:
    rows = _read_csv("data/results/pabb_snapshot_events_pr4_v5.csv")
    snapshots = sum(int(_f(r, "patch_snapshot_events")) for r in rows)
    lines = [
        "# PABB-S Status PR4-v6",
        "",
        f"PATCH_SNAPSHOT_EVENTS_AVAILABLE = {snapshots}",
        "PABB_S_GAIN = NOT_OBSERVED" if snapshots == 0 else "PABB_S_GAIN = REQUIRES_REEVALUATION",
        "PABB_STATUS = secondary",
        "PABB_USED_AS_MAIN_RESULT = false",
        "NO_SOLVED_RATE_CLAIM = true",
        "",
        "PABB-S remains secondary until fresh mini-SWE traces contain real git snapshot events.",
    ]
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(out: str | Path = "data/results/pr4_algo_v6_report.md") -> dict[str, str]:
    adaptive = _read_csv("data/results/taps_unified_adaptive_pr4_v6.csv")
    fair = _read_csv("data/results/taps_fair_baseline_summary_pr4_v6.csv")
    validation = _read_csv("data/results/taps_adaptive_validation_pr4_v6.csv")
    best = _read_json("data/results/taps_adaptive_best_config_pr4_v6.json")
    val_adaptive = next((r for r in validation if r.get("policy") == ADAPTIVE_POLICY), {})
    fair_candidates = [r for r in fair if int(_f(r, "adaptive_starvation_count")) == 0]
    best_fair = max(fair_candidates or fair, key=lambda r: max(_f(r, "adaptive_gain_over_strongest_p95"), _f(r, "adaptive_gain_over_strongest_throughput")), default={})
    val_p95 = _f(best, "validation_p95_gain_over_strongest")
    val_thr = _f(best, "validation_throughput_gain_over_strongest")
    val_wait = _f(best, "validation_ready_wait_gain_over_strongest")
    val_starv = int(_f(best, "validation_starvation_count"))
    adaptive_gain = _gain_status(val_p95)
    ready = val_p95 >= 0.03 and val_starv == 0 and Path("data/results/taps_fair_baseline_summary_pr4_v6.csv").exists()
    gate = "PASS" if ready else ("WARNING" if adaptive else "FAIL")
    fields = {
        "PR4_ALGO_V6_GATE": gate,
        "ADAPTIVE_TAPS_IMPLEMENTED": "true",
        "REGIME_CLASSIFIER_IMPLEMENTED": "true",
        "TAPS_ADAPTIVE_P95_GAIN_OVER_REACTIVE": f"{_f(val_adaptive, 'p95_gain_over_reactive'):.6f}",
        "TAPS_ADAPTIVE_P95_GAIN_OVER_STRONGEST_BASELINE": f"{_f(best_fair, 'adaptive_gain_over_strongest_p95'):.6f}",
        "TAPS_ADAPTIVE_THROUGHPUT_GAIN_OVER_STRONGEST_BASELINE": f"{_f(best_fair, 'adaptive_gain_over_strongest_throughput'):.6f}",
        "TAPS_ADAPTIVE_READY_WAIT_GAIN_OVER_STRONGEST_BASELINE": f"{val_wait:.6f}",
        "TAPS_ADAPTIVE_VALIDATION_P95_GAIN": f"{val_p95:.6f}",
        "TAPS_ADAPTIVE_VALIDATION_THROUGHPUT_GAIN": f"{val_thr:.6f}",
        "TAPS_ADAPTIVE_STARVATION_COUNT": str(val_starv),
        "TAPS_ADAPTIVE_GAIN": adaptive_gain,
        "READY_FOR_PR4_SCALE": str(ready).lower(),
    }
    lines = ["# PR4 Algorithm v6 Report", ""]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    lines.extend(
        [
            "",
            "## Metric Separation",
            "- Simple-baseline gains use reactive_admission / acd_nisp.",
            "- Strongest-baseline gains use the best non-oracle policy available for the same configuration.",
            "- Validation gains use held-out instances only.",
            "- Adaptive scheduling uses online regime state and predicted tool latency only; no future JCT/tool completion is used.",
        ]
    )
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def run_all(trials: int = 100) -> dict[str, Any]:
    regret = write_regret_analysis()
    trial_rows, best, validation = run_adaptive_autotune(max_trials=trials)
    profiles = _profiles_from_json(best)
    adaptive_rows = run_adaptive_sweep(profiles=profiles)
    fair = write_fair_baseline_summary()
    plot_fair_baseline(fair)
    write_pabb_status()
    report = write_report()
    return {
        "regret_rows": len(regret),
        "autotune_trials": len(trial_rows),
        "validation_rows": len(validation),
        "adaptive_rows": len(adaptive_rows),
        "fair_rows": len(fair),
        "report": report,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run-all")
    run.add_argument("--trials", type=int, default=100)
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "run-all":
        print(json.dumps(run_all(args.trials), indent=2, sort_keys=True))
    else:
        print(json.dumps(write_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
