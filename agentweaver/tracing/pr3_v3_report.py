from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _bool(text: Any) -> bool:
    return str(text).strip().lower() in {"true", "1", "yes", "pass", "completed"}


def _int(text: Any, default: int = 0) -> int:
    try:
        if text in {"", None}:
            return default
        return int(float(str(text)))
    except (TypeError, ValueError):
        return default


def _float(text: Any, default: float = 0.0) -> float:
    try:
        if text in {"", None}:
            return default
        return float(str(text))
    except (TypeError, ValueError):
        return default


def _patch_status(path: Path) -> tuple[str, int, str]:
    rows = _read_csv(path)
    if not rows:
        return "FAIL", 0, "patch extraction report missing or empty"
    extracted = sum(1 for r in rows if _bool(r.get("patch_extracted")))
    if extracted > 0:
        return "PASS", extracted, ""
    reason = rows[0].get("reason", "no patch extracted")
    return "FAIL", 0, f"FAIL_NO_PATCH: {reason}"


def _official(summary: Path) -> dict[str, Any]:
    rows = _read_csv(summary)
    if not rows:
        return {
            "used": False,
            "status": "MISSING",
            "evaluated": 0,
            "passed": 0,
            "failed": 0,
            "message": "official eval summary missing",
        }
    row = rows[-1]
    return {
        "used": _bool(row.get("official_verifier_used")),
        "status": row.get("status", ""),
        "evaluated": _int(row.get("official_verifier_num_evaluated")),
        "passed": _int(row.get("official_verifier_num_pass")),
        "failed": _int(row.get("official_verifier_num_fail")),
        "message": row.get("message", ""),
    }


def _unknown_verifier_after_merge(results: Path) -> int:
    verified = results / "mini_swe_lite5_timed_verified_trace_summary.csv"
    if verified.exists():
        return sum(_int(r.get("unknown_verifier_results")) for r in _read_csv(verified))
    return sum(_int(r.get("unknown_verifier_results")) for r in _read_csv(results / "mini_swe_lite5_timed_trace_summary.csv")) + sum(
        _int(r.get("unknown_verifier_results")) for r in _read_csv(results / "mini_swe_lite10_r4_timed_trace_summary.csv")
    )


def _bes_status(path: Path) -> tuple[str, str, float, int]:
    rows = _read_csv(path)
    if not rows:
        return "FAIL", "WARNING", 0.0, 0
    configs = {(r.get("mesh"), r.get("effective_regions")) for r in rows}
    gains = [_float(r.get("acd_bes_gain_vs_acd_only")) for r in rows if r.get("policy") == "acd_bes"]
    max_gain = max(gains) if gains else 0.0
    effect = "OBSERVED" if max_gain > 1e-6 else "NOT_OBSERVED"
    return ("PASS" if len(configs) >= 3 else "FAIL", effect, max_gain, len(configs))


def _breakdown_status(path: Path) -> tuple[str, float, float]:
    rows = _read_csv(path)
    if not rows:
        return "FAIL", 0.0, 0.0
    jct = sum(_float(r.get("measured_agent_jct")) for r in rows)
    tool = sum(_float(r.get("measured_tool_wall_time")) for r in rows)
    llm = sum(_float(r.get("measured_llm_wall_time")) for r in rows)
    return "PASS", (tool / jct if jct > 0 else 0.0), (llm / jct if jct > 0 else 0.0)


def generate_report(results_dir: str | Path = "data/results", out: str | Path = "data/results/pr3_v3_report.md") -> dict[str, Any]:
    results = Path(results_dir)
    patch_status, patch_count, patch_reason = _patch_status(results / "mini_swe_lite5_timed_patch_extraction_report.csv")
    official = _official(results / "mini_swe_lite5_timed_official_eval_summary.csv")
    bes_status, bes_effect, bes_gain, bes_configs = _bes_status(results / "bes_stress_mini_swe_pr3_v3.csv")
    breakdown_status, tool_share, llm_share = _breakdown_status(results / "mini_swe_lite10_r4_timed_latency_breakdown_detailed.csv")
    unknown_after = _unknown_verifier_after_merge(results)
    official_clear = official["used"] and official["evaluated"] >= 1
    official_clear = official_clear or (not official["used"] and bool(official["message"]))
    ready = (
        Path("data/traces/mini_swe_lite10_r4_timed").exists()
        and bes_status in {"PASS", "WARNING"}
        and breakdown_status == "PASS"
        and official_clear
    )
    gate = "PASS" if patch_status == "PASS" and official["used"] and bes_status == "PASS" and breakdown_status == "PASS" else "WARNING"
    if bes_status == "FAIL" or breakdown_status == "FAIL" or not official_clear:
        gate = "FAIL"
    fields: dict[str, Any] = {
        "PR3_V3_GATE": gate,
        "PATCH_EXTRACTION": patch_status,
        "PATCH_EXTRACTION_NUM_PREDICTIONS": patch_count,
        "PATCH_EXTRACTION_REASON": patch_reason,
        "OFFICIAL_VERIFIER_USED": str(bool(official["used"])).lower(),
        "OFFICIAL_VERIFIER_STATUS": official["status"],
        "OFFICIAL_VERIFIER_NUM_EVALUATED": official["evaluated"],
        "OFFICIAL_VERIFIER_NUM_PASS": official["passed"],
        "OFFICIAL_VERIFIER_NUM_FAIL": official["failed"],
        "OFFICIAL_VERIFIER_MESSAGE": official["message"],
        "UNKNOWN_VERIFIER_RESULTS_AFTER_MERGE": unknown_after,
        "BES_STRESS_EVALUATION": bes_status,
        "BES_REAL_TRACE_EFFECT": bes_effect,
        "BES_MAX_GAIN_VS_ACD_ONLY": f"{bes_gain:.6f}",
        "BES_RESOURCE_CONFIGURATIONS": bes_configs,
        "MODEL_TOOL_BREAKDOWN": breakdown_status,
        "MEASURED_TOOL_TIME_SHARE_LITE10_R4": f"{tool_share:.6f}",
        "MEASURED_LLM_TIME_SHARE_LITE10_R4": f"{llm_share:.6f}",
        "READY_FOR_PR4_PILOT": str(ready).lower(),
    }
    text = ["# PR3-v3 Report", ""]
    for key, value in fields.items():
        text.append(f"{key} = {value}")
    text.extend(
        [
            "",
            "Notes:",
            "- Official SWE-bench verifier results are only merged when the harness evaluates non-empty patches.",
            "- Missing patches remain unknown and are not counted as pass or fail.",
            "- BES stress uses constrained effective compute regions; any reported gain is measured by replay, not hardcoded.",
            "- Measured wall-clock tool/LLM time is reported separately from simulated H100 model-side replay time.",
        ]
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(text) + "\n", encoding="utf-8")
    return fields


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="data/results")
    ap.add_argument("--out", default="data/results/pr3_v3_report.md")
    args = ap.parse_args()
    fields = generate_report(args.results_dir, args.out)
    print("\n".join(f"{k} = {v}" for k, v in fields.items()))


if __name__ == "__main__":
    main()
