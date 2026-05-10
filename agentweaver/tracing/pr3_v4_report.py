from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass", "completed"}


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _fields_from_md(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    if not path.exists():
        return fields
    for line in path.read_text(encoding="utf-8").splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            fields[key.strip()] = value.strip()
    return fields


def _official(summary: Path) -> dict[str, Any]:
    rows = _read_csv(summary)
    if not rows:
        return {"used": False, "evaluated": 0, "passed": 0, "failed": 0, "message": "official eval summary missing"}
    row = rows[-1]
    return {
        "used": _bool(row.get("official_verifier_used")),
        "evaluated": _int(row.get("official_verifier_num_evaluated")),
        "passed": _int(row.get("official_verifier_num_pass")),
        "failed": _int(row.get("official_verifier_num_fail")),
        "message": row.get("message", ""),
    }


def _unknown_after_merge(results: Path) -> int:
    verified = results / "mini_swe_lite5_patchcap_verified_trace_summary.csv"
    if verified.exists():
        return sum(_int(r.get("unknown_verifier_results")) for r in _read_csv(verified))
    summary = results / "mini_swe_lite5_patchcap_trace_summary.csv"
    if summary.exists():
        return sum(_int(r.get("unknown_verifier_results")) for r in _read_csv(summary))
    return 0


def generate_report(results_dir: str | Path = "data/results", out: str | Path = "data/results/pr3_v4_report.md") -> dict[str, Any]:
    results = Path(results_dir)
    patch_rows = _read_csv(results / "mini_swe_lite5_patchcap_patch_extraction_report.csv")
    patch_predictions = sum(1 for r in patch_rows if _bool(r.get("patch_extracted")))
    debug_rows = _read_csv(results / "patch_capture_debug_pr3_v4.csv")
    attempted = bool(debug_rows)
    empty_count = sum(1 for r in debug_rows if _bool(r.get("patch_empty")))
    error_count = sum(1 for r in debug_rows if str(r.get("patch_capture_error", "")).strip())
    official = _official(results / "mini_swe_lite5_patchcap_official_eval_summary.csv")
    bes = _fields_from_md(results / "bes_positioning_pr3_v4.md")
    timed_valid = (results / "mini_swe_lite5_patchcap_trace_summary.csv").exists()
    clear_official_state = (official["used"] and official["evaluated"] >= 1) or (
        not official["used"] and patch_predictions == 0 and "empty" in str(official["message"]).lower()
    )
    patch_capture = "PASS" if attempted and error_count == 0 and empty_count < len(debug_rows) else "WARNING"
    if not attempted:
        patch_capture = "FAIL"
    ready = timed_valid and attempted and clear_official_state and Path("data/results/bes_positioning_pr3_v4.md").exists()
    if official["used"] and official["evaluated"] >= 1 and patch_capture in {"PASS", "WARNING"}:
        gate = "PASS"
    elif ready:
        gate = "WARNING"
    else:
        gate = "FAIL"
    fields: dict[str, Any] = {
        "PR3_V4_GATE": gate,
        "PATCH_CAPTURE": patch_capture,
        "PATCH_EXTRACTION_NUM_PREDICTIONS": patch_predictions,
        "OFFICIAL_VERIFIER_USED": str(bool(official["used"])).lower(),
        "OFFICIAL_VERIFIER_NUM_EVALUATED": official["evaluated"],
        "OFFICIAL_VERIFIER_NUM_PASS": official["passed"],
        "OFFICIAL_VERIFIER_NUM_FAIL": official["failed"],
        "UNKNOWN_VERIFIER_RESULTS_AFTER_MERGE": _unknown_after_merge(results),
        "PATCH_CAPTURE_EMPTY_COUNT": empty_count,
        "PATCH_CAPTURE_ERROR_COUNT": error_count,
        "BES_REAL_TRACE_EFFECT": bes.get("BES_REAL_TRACE_EFFECT", "NOT_OBSERVED"),
        "READY_FOR_PR4_PILOT": str(bool(ready)).lower(),
        "OFFICIAL_VERIFIER_MESSAGE": official["message"],
    }
    lines = ["# PR3-v4 Report", ""]
    lines.extend(f"{key} = {value}" for key, value in fields.items())
    lines.extend(
        [
            "",
            "Notes:",
            "- Empty patches are not written to predictions.",
            "- Missing or unevaluated instances remain unknown, not fail/pass.",
            "- Official verifier counts come only from the SWE-bench harness summary.",
        ]
    )
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="data/results")
    ap.add_argument("--out", default="data/results/pr3_v4_report.md")
    args = ap.parse_args()
    fields = generate_report(args.results_dir, args.out)
    print("\n".join(f"{k} = {v}" for k, v in fields.items()))


if __name__ == "__main__":
    main()
