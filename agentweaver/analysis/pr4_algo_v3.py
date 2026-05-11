from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentweaver.analysis.pr4_algo_diagnosis import write_diagnosis
from agentweaver.simulator.context_domain_factorization import compare_strict_prefix_reuse
from agentweaver.simulator.multisession_replay import TAPSConfig, run_taps_v3_sweep
from agentweaver.simulator.pabb_online_replay import run_pabb_online_v3
from agentweaver.simulator.taps_autotune import run_taps_autotune
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
        value = row.get(key, "")
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def _gain(base: float, new: float, lower_better: bool = True) -> float:
    if base <= 0:
        return 0.0
    return (base - new) / base if lower_better else (new - base) / base


def write_cdf_limitations(
    cdf_csv: str | Path = "data/results/cdf_strict_prefix_comparison_pr4_v2.csv",
    out: str | Path = "data/results/cdf_limitations_pr4_v3.md",
) -> None:
    rows = _read_csv(cdf_csv)
    added = sum(_f(r, "cdf_added_reusable_tokens") for r in rows)
    natural = sum(_f(r, "natural_strict_prefix_reusable_tokens") for r in rows)
    cdf = sum(_f(r, "cdf_canonical_prefix_reusable_tokens") for r in rows)
    total = sum(_f(r, "segment_reuse_potential_tokens") for r in rows)
    block = any(str(r.get("block_prefix_mode", "")).lower() == "true" for r in rows)
    text = f"""# CDF Limitations for PR4-algo-v3

raw_token_ids_available = false
block_prefix_mode = {str(block).lower()}
natural_strict_prefix_reusable_tokens = {int(natural)}
cdf_canonical_prefix_reusable_tokens = {int(cdf)}
cdf_added_reusable_tokens = {int(added)}
segment_reuse_potential_tokens = {int(total)}

The current mini-SWE traces do not include raw token ids, so strict accounting uses ordered segment token hashes and lengths in block-prefix mode. This is a conservative replay approximation for prefix continuity, not a claim of semantic KV reuse.

The collected traces already have high natural strict/block-prefix reuse because mini-SWE repeatedly renders task, tool, and repository context in a stable order. CDF therefore adds only a small amount of reusable context and is not the main real mini-SWE speedup source in PR4-algo-v3.

CDF remains useful as a prompt-rendering compiler for agent workloads with unstable prompt order or fragmented shared history. In this report it is kept as a secondary mechanism unless stronger non-oracle evidence appears.
"""
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")


def write_mechanism_positioning_v3(out: str | Path = "data/results/mechanism_positioning_pr4_algo_v3.md") -> None:
    text = """# PR4 Algorithm v3 Mechanism Positioning

ACD/NISP:
ACD and NISP remain validated on real timed mini-SWE traces. ACD accounts only for prefix-safe context reuse; NISP accounts for measured tool-stall state parking and resume prefill reduction.

TAPS-v3:
TAPS-v3 is the main scheduling/runtime contribution when validation shows p95 JCT or ready-queue wait improvement under multi-session pressure. The main result uses predictive tool latency from leave-one-instance-out command-class medians plus already observed same-session history. taps_oracle_upper_bound is reported only as an upper bound.

CDF:
CDF uses strict/block-prefix accounting. It is a secondary prompt-rendering compiler in the current mini-SWE data because the observed gain is weak; it is not used as the main performance claim unless new data shows stronger non-oracle gain.

PABB-v3:
PABB-v3 is an event-level online branch-budget mechanism. Its main result may use only signals revealed by executed LLM/tool/verifier/patch events. pabb_oracle_upper_bound is an upper bound and cannot be reported as online performance.

Old BES:
Deprecated. It is not restored as a main mechanism and is not used for real mini-SWE main-result attribution.

Correctness:
No solved-rate claims are made without official verifier results. Unknown verifier outcomes remain unknown.
"""
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")


def _best_config() -> TAPSConfig:
    path = Path("data/results/taps_v3_best_config.json")
    if not path.exists():
        return TAPSConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    return TAPSConfig(**data.get("config", {}))


def _taps_gain_rows(rows: list[dict[str, str]], sessions: int, arrival: str = "bursty", regions: int = 4) -> tuple[float, float, float]:
    base = next(
        (
            r
            for r in rows
            if r.get("policy") == "acd_nisp"
            and int(r.get("sessions", 0) or 0) == sessions
            and str(r.get("arrival_pattern")) == arrival
            and int(r.get("effective_regions", 0) or 0) == regions
        ),
        None,
    )
    taps = next(
        (
            r
            for r in rows
            if r.get("policy") == "taps_v3"
            and int(r.get("sessions", 0) or 0) == sessions
            and str(r.get("arrival_pattern")) == arrival
            and int(r.get("effective_regions", 0) or 0) == regions
        ),
        None,
    )
    return (
        _gain(_f(base, "p95_jct"), _f(taps, "p95_jct")),
        _gain(_f(base, "ready_queue_wait"), _f(taps, "ready_queue_wait")),
        _gain(_f(base, "throughput_sessions_per_sec"), _f(taps, "throughput_sessions_per_sec"), lower_better=False),
    )


def _pabb_summary(rows: list[dict[str, str]]) -> tuple[float, float]:
    gains = [_f(r, "pabb_online_gain_vs_fcfs") for r in rows if r.get("policy") == "pabb_online" and r.get("pabb_online_gain_vs_fcfs") not in {"", None}]
    gaps = [_f(r, "oracle_gap") for r in rows if r.get("policy") == "pabb_online" and r.get("oracle_gap") not in {"", None}]
    return (sum(gains) / len(gains) if gains else 0.0, sum(gaps) / len(gaps) if gaps else 0.0)


def _status_from_gain(gain: float) -> str:
    if gain >= 0.10:
        return "STRONG"
    if gain >= 0.03:
        return "MODERATE"
    if gain > 0:
        return "WEAK"
    return "NOT_OBSERVED"


def run_pabb_tests(out: str | Path = "data/results/pabb_no_future_leakage_tests_pr4_v3.txt") -> str:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        ["python", "-m", "pytest", "-q", "tests/test_pabb_no_future_leakage.py"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return "PASS" if proc.returncode == 0 else "FAIL"


def write_report(
    out: str | Path = "data/results/pr4_algo_v3_report.md",
    test_status: str = "UNKNOWN",
) -> dict[str, str]:
    cdf_rows = _read_csv("data/results/cdf_strict_prefix_comparison_pr4_v2.csv")
    taps_rows = _read_csv("data/results/taps_v3_sweep.csv")
    pabb_rows = _read_csv("data/results/pabb_online_branch_budget_pr4_v3.csv")
    validation = _read_csv("data/results/taps_v3_validation.csv")
    train = _read_csv("data/results/taps_v3_autotune.csv")

    cdf_added = sum(_f(r, "cdf_added_reusable_tokens") for r in cdf_rows)
    cdf_saved = sum(_f(r, "estimated_prefill_saved") for r in cdf_rows)
    cdf_speed = _gain(sum(_f(r, "natural_strict_prefix_reusable_tokens") for r in cdf_rows), sum(_f(r, "cdf_canonical_prefix_reusable_tokens") for r in cdf_rows), lower_better=False)
    p95_16, wait_16, thr_16 = _taps_gain_rows(taps_rows, 16)
    p95_32, _, _ = _taps_gain_rows(taps_rows, 32)
    p95_64, _, _ = _taps_gain_rows(taps_rows, 64)
    taps_status = _status_from_gain(max(p95_16, p95_32, p95_64))
    pabb_gain, pabb_gap = _pabb_summary(pabb_rows)
    pabb_status = _status_from_gain(pabb_gain)
    best_cfg = asdict(_best_config())
    train_best = min((_f(r, "objective", 1e9) for r in train), default=0.0)
    valid_obj = _f(validation[0] if validation else None, "objective")
    overfit = bool(train_best > 0 and valid_obj > train_best * 1.25)
    required = [
        Path("data/results/pr4_algo_v3_diagnosis.md"),
        Path("data/results/taps_v3_sweep.csv"),
        Path("data/results/taps_v3_best_config.json"),
        Path("data/results/taps_v3_validation.csv"),
        Path("data/results/pabb_online_branch_budget_pr4_v3.csv"),
        Path("data/results/cdf_limitations_pr4_v3.md"),
        Path("data/results/mechanism_positioning_pr4_algo_v3.md"),
    ]
    ready = all(p.exists() for p in required) and test_status == "PASS"
    fields = {
        "PR4_ALGO_V3_GATE": "PASS" if ready else "WARNING",
        "CDF_GAIN": "WEAK" if cdf_added > 0 and cdf_speed < 0.01 else ("OBSERVED" if cdf_added > 0 else "NOT_OBSERVED"),
        "CDF_ADDED_REUSABLE_TOKENS": str(int(cdf_added)),
        "CDF_USED_AS_MAIN_RESULT": "false",
        "TAPS_V3_GAIN": taps_status,
        "TAPS_V3_BEST_CONFIG": json.dumps(best_cfg, sort_keys=True),
        "TAPS_V3_P95_GAIN_AT_16": f"{p95_16:.6f}",
        "TAPS_V3_P95_GAIN_AT_32": f"{p95_32:.6f}",
        "TAPS_V3_P95_GAIN_AT_64": f"{p95_64:.6f}",
        "TAPS_V3_READY_WAIT_GAIN_AT_16": f"{wait_16:.6f}",
        "TAPS_V3_THROUGHPUT_GAIN_AT_16": f"{thr_16:.6f}",
        "TAPS_V3_VALIDATION_OVERFIT": str(overfit).lower(),
        "PABB_ONLINE_IMPLEMENTED": str(Path("data/results/pabb_online_branch_budget_pr4_v3.csv").exists()).lower(),
        "PABB_NO_FUTURE_LEAKAGE_TESTS": test_status,
        "PABB_ONLINE_GAIN": pabb_status,
        "PABB_ORACLE_GAP": f"{pabb_gap:.6f}",
        "PABB_USED_AS_MAIN_RESULT": str(pabb_status in {"STRONG", "MODERATE"}).lower(),
        "OLD_BES_DEPRECATED": "true",
        "REAL_MINISWE_MAIN_MECHANISMS": "ACD/NISP/TAPS",
        "CDF_STATUS": "optional",
        "PABB_STATUS": "main" if pabb_status in {"STRONG", "MODERATE"} else "optional",
        "READY_FOR_PR4_SCALE": str(ready).lower(),
    }
    lines = ["# PR4 Algorithm v3 Report", ""]
    lines.extend(f"{k} = {v}" for k, v in fields.items())
    lines.extend(
        [
            "",
            "## Evidence Boundaries",
            "- TAPS-v3 numbers are predictive, non-oracle replay results unless explicitly named taps_oracle_upper_bound.",
            "- PABB-v3 online uses only event-prefix-visible signals; oracle rows are upper bounds.",
            "- CDF is strict/block-prefix replay accounting and remains secondary for the current mini-SWE traces.",
            "- Metrics are replay/model-side or measured-tool-side as labeled in their CSVs; no solved-rate claim is made without official verifier coverage.",
        ]
    )
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def run_all(max_configs: int = 64) -> dict[str, Any]:
    compare_strict_prefix_reuse()
    diagnosis = write_diagnosis()
    autotune_rows, best, validation = run_taps_autotune(max_configs=max_configs)
    cfg = TAPSConfig(**best["config"])
    taps_rows = run_taps_v3_sweep(taps_config=cfg)
    pabb_rows = run_pabb_online_v3()
    write_cdf_limitations()
    write_mechanism_positioning_v3()
    test_status = run_pabb_tests()
    report = write_report(test_status=test_status)
    return {
        "diagnosis": diagnosis,
        "autotune_rows": len(autotune_rows),
        "validation_rows": len(validation),
        "taps_rows": len(taps_rows),
        "pabb_rows": len(pabb_rows),
        "pabb_tests": test_status,
        "report": report,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run-all")
    run.add_argument("--max-configs", type=int, default=64)
    sub.add_parser("diagnosis")
    sub.add_parser("report")
    sub.add_parser("cdf-limitations")
    sub.add_parser("positioning")
    sub.add_parser("test-pabb")
    args = ap.parse_args()
    if args.cmd == "run-all":
        print(json.dumps(run_all(args.max_configs), indent=2, sort_keys=True))
    elif args.cmd == "diagnosis":
        print(json.dumps(write_diagnosis(), indent=2, sort_keys=True))
    elif args.cmd == "report":
        print(json.dumps(write_report(), indent=2, sort_keys=True))
    elif args.cmd == "cdf-limitations":
        write_cdf_limitations()
        print(json.dumps({"out": "data/results/cdf_limitations_pr4_v3.md"}, indent=2))
    elif args.cmd == "positioning":
        write_mechanism_positioning_v3()
        print(json.dumps({"out": "data/results/mechanism_positioning_pr4_algo_v3.md"}, indent=2))
    elif args.cmd == "test-pabb":
        print(json.dumps({"PABB_NO_FUTURE_LEAKAGE_TESTS": run_pabb_tests()}, indent=2))


if __name__ == "__main__":
    main()
