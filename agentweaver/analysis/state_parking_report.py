from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agentweaver.simulator.state_parking_modes import estimate_nisp_private_metrics, state_parking_rows
from agentweaver.utils.io import ensure_dir, write_csv


def write_state_parking_report(
    schedule_summary_csv: str | Path = "data/results/schedule_summary_pr4_v12.csv",
    out_csv: str | Path = "data/results/nisp_state_parking_pr4_v13.csv",
    out_md: str | Path = "data/results/nisp_state_parking_summary_pr4_v13.md",
) -> dict[str, Any]:
    rows = state_parking_rows(schedule_summary_csv)
    write_csv(out_csv, rows)
    metrics = estimate_nisp_private_metrics(schedule_summary_csv)
    counts = Counter(str(r["state_residency"]) for r in rows)
    fields: dict[str, Any] = {
        "NISP_STATE_PARKING_ROWS": len(rows),
        "NISP_STATE_PARKING_STATUS": "PASS" if rows else "FAIL",
        "HOT_STALLS": counts.get("HOT", 0),
        "WARM_STALLS": counts.get("WARM", 0),
        "COLD_STALLS": counts.get("COLD", 0),
        "NONE_STALLS": counts.get("NONE", 0),
        "PARKED_STATE_BYTES": int(metrics["parked_state_bytes"]),
        "PRIVATE_SUFFIX_HIT_TOKENS": int(metrics["private_suffix_hit_tokens"]),
        "RESUME_PREFILL_TOKENS_SAVED_BY_NISP": int(metrics["resume_prefill_tokens_saved"]),
    }
    lines = ["# NISP State Parking PR4-v13", ""]
    lines.extend(f"{key} = {value}" for key, value in fields.items())
    lines.extend(
        [
            "",
            "## Decision Model",
            "NISP parks only branch-private state across tool stalls. Shared prefix cache hits are excluded here so ACD and NISP are not double-counted.",
            "HOT/WARM/COLD decisions are attributed from schedule state_residency and parked_state_bytes. If most states are WARM/COLD, the reason is memory pressure relative to expected reuse and stall duration.",
            "",
            "NISP utility = predicted_reuse_probability * recompute_cost_saved; cost = parked_state_bytes * memory_pressure * stall_duration.",
        ]
    )
    p = Path(out_md)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default="data/results/schedule_summary_pr4_v12.csv")
    ap.add_argument("--out", default="data/results/nisp_state_parking_pr4_v13.csv")
    ap.add_argument("--summary", default="data/results/nisp_state_parking_summary_pr4_v13.md")
    args = ap.parse_args()
    print(json.dumps(write_state_parking_report(args.schedule, args.out, args.summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
