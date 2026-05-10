from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.profiling.pr2_v2 import REQUIRED_REAL_POLICIES
from agentweaver.utils.io import ensure_dir, write_csv


SYNTHETIC_SCENARIOS = [
    "S1_context_heavy",
    "S2_branch_heavy",
    "S3_tool_stall_heavy",
    "S4_low_reuse_negative",
    "S5_tool_dominated_negative",
]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any] | None, key: str, default: float = math.nan) -> float:
    if not row:
        return default
    try:
        value = row.get(key, "")
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def _parse_kv(path: str | Path) -> dict[str, str]:
    p = Path(path)
    out: dict[str, str] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if " = " in line:
            k, v = line.split(" = ", 1)
            out[k.strip()] = v.strip()
    return out


def _prefix_latency_comparison(rows: list[dict[str, str]]) -> tuple[str, str]:
    if not rows:
        return "FAIL", "missing prefix reuse effect CSV"
    if any(str(row.get("prefix_metrics_reliable", "")).lower() == "true" for row in rows):
        return "FAIL", "prefix_reuse_effect contains prefix_metrics_reliable=true"
    if any(str(row.get("prefix_cache_counters_used_as_evidence", "false")).lower() == "true" for row in rows):
        return "FAIL", "prefix_reuse_effect uses cache counters as evidence"
    shared_rows = [r for r in rows if _float(r, "shared_prefix_tokens", 0) > 0]
    if not shared_rows:
        return "FAIL", "no shared-prefix latency comparison rows"
    better = [r for r in shared_rows if _float(r, "latency_reduction", -1) > 0]
    if len(better) == len(shared_rows):
        return "PASS", f"{len(better)}/{len(shared_rows)} shared-prefix cases have lower latency on prefix server"
    if better:
        return "WARNING", f"{len(better)}/{len(shared_rows)} shared-prefix cases have lower latency on prefix server"
    return "WARNING", "prefix-enabled server was not faster for shared-prefix cases"


def plot_prefix_benefit(effect_csv: str | Path, out: str | Path) -> None:
    rows = _read_csv(effect_csv)
    rows = [r for r in rows if _float(r, "shared_prefix_tokens", 0) > 0]
    if not rows:
        raise RuntimeError(f"no shared-prefix rows in {effect_csv}")
    suffixes = sorted({int(_float(r, "unique_suffix_tokens", 0)) for r in rows})
    fig, axes = plt.subplots(1, len(suffixes), figsize=(4.4 * len(suffixes), 3.6), sharey=True)
    if len(suffixes) == 1:
        axes = [axes]
    for ax, suffix in zip(axes, suffixes):
        sub = [r for r in rows if int(_float(r, "unique_suffix_tokens", 0)) == suffix]
        for num_requests in sorted({int(_float(r, "num_requests", 0)) for r in sub}):
            curve = sorted(
                [r for r in sub if int(_float(r, "num_requests", 0)) == num_requests],
                key=lambda r: _float(r, "shared_prefix_tokens", 0),
            )
            ax.plot(
                [_float(r, "shared_prefix_tokens", 0) for r in curve],
                [100.0 * _float(r, "latency_reduction", 0) for r in curve],
                marker="o",
                linewidth=1.8,
                label=f"num_requests={num_requests}",
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"unique suffix {suffix}")
        ax.set_xlabel("shared_prefix_tokens")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("latency_reduction (%)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    ensure_dir(Path(out).parent)
    fig.savefig(out)
    plt.close(fig)


def _by_scenario_policy(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(r.get("scenario", ""), r.get("policy", "")): r for r in rows}


def synthetic_status(summary_rows: list[dict[str, str]]) -> tuple[str, str, dict[str, float]]:
    if not summary_rows:
        return "FAIL", "missing H100-calibrated synthetic replay summary", {}
    table = _by_scenario_policy(summary_rows)
    missing: list[str] = []
    for scenario in SYNTHETIC_SCENARIOS:
        for policy in REQUIRED_REAL_POLICIES:
            if (scenario, policy) not in table:
                missing.append(f"{scenario}:{policy}")
    if ("AGGREGATE", "full_agentweaver") not in table:
        missing.append("AGGREGATE:full_agentweaver")
    if missing:
        return "FAIL", "missing rows " + "; ".join(missing[:12]), {}

    benefits: dict[str, float] = {}
    not_worse: list[str] = []
    for scenario in SYNTHETIC_SCENARIOS:
        naive = table[(scenario, "naive_wafer")]
        full = table[(scenario, "full_agentweaver")]
        naive_jct = _float(naive, "jct")
        full_jct = _float(full, "jct")
        benefit = (naive_jct - full_jct) / max(1e-9, naive_jct)
        benefits[scenario] = benefit
        if scenario in {"S1_context_heavy", "S2_branch_heavy", "S3_tool_stall_heavy"} and full_jct <= naive_jct * 1.001:
            not_worse.append(scenario)

    positive = [benefits[s] for s in ("S1_context_heavy", "S2_branch_heavy", "S3_tool_stall_heavy")]
    pos_mean = sum(positive) / max(1, len(positive))
    neg = [benefits[s] for s in ("S4_low_reuse_negative", "S5_tool_dominated_negative")]
    negative_reduced = all(x <= max(0.0, pos_mean) * 0.5 for x in neg)

    agg_naive = table[("AGGREGATE", "naive_wafer")]
    agg_acd = table[("AGGREGATE", "acd_only")]
    agg_nisp = table[("AGGREGATE", "acd_nisp")]
    agg_full = table[("AGGREGATE", "full_agentweaver")]
    direction_ok = (
        _float(agg_acd, "prefill_tokens_avoided", 0) > _float(agg_naive, "prefill_tokens_avoided", 0)
        and _float(agg_nisp, "resume_prefill_tokens", math.inf) <= _float(agg_naive, "resume_prefill_tokens", math.inf)
        and _float(agg_full, "branch_wasted_tokens", math.inf) <= _float(agg_naive, "branch_wasted_tokens", math.inf)
    )

    notes = [
        f"S1/S2/S3_not_worse={len(not_worse)}/3",
        f"positive_mean_benefit={pos_mean:.6f}",
        f"S4_benefit={benefits['S4_low_reuse_negative']:.6f}",
        f"S5_benefit={benefits['S5_tool_dominated_negative']:.6f}",
        f"negative_controls_reduced={negative_reduced}",
        f"ablation_direction_ok={direction_ok}",
    ]
    if len(not_worse) == 3 and negative_reduced and direction_ok:
        return "PASS", "; ".join(notes), benefits
    return "WARNING", "; ".join(notes), benefits


def real_all_policies_status(rows: list[dict[str, str]]) -> tuple[str, str]:
    if not rows:
        return "FAIL", "missing real agent-like all-policy CSV"
    by_instance: dict[str, set[str]] = {}
    for row in rows:
        by_instance.setdefault(row.get("instance_id", ""), set()).add(row.get("policy", ""))
    required = set(REQUIRED_REAL_POLICIES)
    missing = [
        f"{instance}:{sorted(required - policies)}"
        for instance, policies in sorted(by_instance.items())
        if instance and not required.issubset(policies)
    ]
    if "AGGREGATE" not in by_instance:
        missing.append("AGGREGATE row missing")
    if missing:
        return "FAIL", "; ".join(missing[:10])
    return "PASS", f"{len(by_instance) - 1} instances plus aggregate include all policies"


def write_final_report(args: argparse.Namespace) -> dict[str, str]:
    prefix_rows = _read_csv(args.prefix_effect_csv)
    prefix_status, prefix_note = _prefix_latency_comparison(prefix_rows)
    holdout = _parse_kv(args.holdout_md)
    holdout_status = holdout.get("HOLDOUT_LATENCY_MODEL_QUALITY", "FAIL")
    synthetic_rows = _read_csv(args.synthetic_summary_csv)
    synthetic_replay_status, synthetic_note, benefits = synthetic_status(synthetic_rows)
    real_status, real_note = real_all_policies_status(_read_csv(args.real_all_policies_csv))
    readme = Path(args.readme)
    readme_pr3 = readme.exists() and "PR3: mini-SWE-agent / SWE-agent trace collection plan" in readme.read_text(
        encoding="utf-8"
    )
    scripts_exist = Path("scripts/run_mini_swe_trace_pr3.sh").exists() and Path(
        "scripts/run_mini_swe_multibranch_pr3.sh"
    ).exists()
    prefix_fixed = (
        prefix_status != "FAIL"
        and bool(prefix_rows)
        and not any(str(r.get("prefix_metrics_reliable", "")).lower() == "true" for r in prefix_rows)
    )
    holdout_exists = Path(args.holdout_md).exists() and Path(args.holdout_csv).exists()
    synthetic_exists = Path(args.synthetic_summary_csv).exists() and Path(args.ablation_csv).exists()
    real_exists = Path(args.real_all_policies_csv).exists()
    ready = (
        prefix_fixed
        and holdout_exists
        and holdout_status != "FAIL"
        and synthetic_exists
        and synthetic_replay_status != "FAIL"
        and real_exists
        and real_status == "PASS"
        and readme_pr3
        and scripts_exist
    )
    statuses = [prefix_status, holdout_status, synthetic_replay_status, real_status]
    if any(s == "FAIL" for s in statuses) or not ready:
        gate = "FAIL" if any(s == "FAIL" for s in statuses) else "WARNING"
    else:
        gate = "WARNING" if any(s == "WARNING" for s in statuses) else "PASS"

    fields = {
        "PR2_V2_1_GATE": gate,
        "PREFIX_CACHE_METRICS_RELIABLE": "false",
        "PREFIX_CACHE_COUNTERS_USED_AS_EVIDENCE": "false",
        "PREFIX_REUSE_LATENCY_COMPARISON": prefix_status,
        "HOLDOUT_LATENCY_MODEL_QUALITY": holdout_status,
        "SYNTHETIC_REPLAY_H100CALIB": synthetic_replay_status,
        "REAL_AGENTLIKE_ALL_POLICIES": real_status,
        "READY_FOR_PR3": str(ready).lower(),
        "CONTROLLED_REAL_AGENTLIKE_NOTE": "real_agentlike_h100 is a controlled pseudo workload, not SWE-bench",
        "random_split_median_ape": holdout.get("random_split_median_ape", ""),
        "random_split_p95_ape": holdout.get("random_split_p95_ape", ""),
        "leave_input_median_ape": holdout.get("leave_input_median_ape", ""),
        "leave_input_p95_ape": holdout.get("leave_input_p95_ape", ""),
        "leave_output_median_ape": holdout.get("leave_output_median_ape", ""),
        "leave_output_p95_ape": holdout.get("leave_output_p95_ape", ""),
        "worst_bucket": holdout.get("worst_bucket", ""),
    }
    notes = {
        "PREFIX_REUSE_LATENCY_COMPARISON_NOTE": prefix_note,
        "SYNTHETIC_REPLAY_H100CALIB_NOTE": synthetic_note,
        "REAL_AGENTLIKE_ALL_POLICIES_NOTE": real_note,
        "README_PR3_SECTION": str(readme_pr3).lower(),
        "PR3_SCRIPTS_EXIST": str(scripts_exist).lower(),
    }
    for scenario, benefit in benefits.items():
        notes[f"{scenario}_full_vs_naive_jct_reduction"] = f"{benefit:.6f}"
    lines = ["# PR2-v2.1 Report", ""]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    lines.append("")
    lines.append("## Notes")
    lines.extend(f"{k} = {v}" for k, v in notes.items())
    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_plot = sub.add_parser("plot-prefix-benefit")
    p_plot.add_argument("--effect-csv", default="data/results/prefix_reuse_effect_pr2_v2_1.csv")
    p_plot.add_argument("--out", default="data/plots/prefix_reuse_latency_benefit_pr2_v2_1.pdf")

    p_report = sub.add_parser("report")
    p_report.add_argument("--out", default="data/results/pr2_v2_1_report.md")
    p_report.add_argument("--prefix-effect-csv", default="data/results/prefix_reuse_effect_pr2_v2_1.csv")
    p_report.add_argument("--holdout-csv", default="data/results/h100_latency_holdout_report_pr2_v2_1.csv")
    p_report.add_argument("--holdout-md", default="data/results/h100_latency_holdout_report_pr2_v2_1.md")
    p_report.add_argument("--synthetic-summary-csv", default="data/results/wafer_replay_summary_h100calib_pr2_v2_1.csv")
    p_report.add_argument("--ablation-csv", default="data/results/ablation_h100calib_pr2_v2_1.csv")
    p_report.add_argument("--real-all-policies-csv", default="data/results/real_agentlike_replay_all_policies_pr2_v2.csv")
    p_report.add_argument("--readme", default="README.md")

    args = ap.parse_args()
    if args.cmd == "plot-prefix-benefit":
        plot_prefix_benefit(args.effect_csv, args.out)
        print(json.dumps({"out": args.out}, indent=2))
    elif args.cmd == "report":
        fields = write_final_report(args)
        print(json.dumps(fields, indent=2))


if __name__ == "__main__":
    main()
