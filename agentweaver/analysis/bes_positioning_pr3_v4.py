from __future__ import annotations

import argparse
from pathlib import Path


def write_bes_positioning(out: str | Path = "data/results/bes_positioning_pr3_v4.md") -> dict[str, str]:
    fields = {
        "BES_REAL_TRACE_EFFECT": "NOT_OBSERVED",
        "BES_USED_FOR_REAL_MINISWE_MAIN_RESULT": "false",
        "BES_RETAINED_FOR_SYNTHETIC_BRANCH_HEAVY": "true",
        "ACD_NISP_USED_FOR_REAL_MINISWE_MAIN_RESULT": "true",
    }
    lines = ["# BES Positioning for PR3-v4", ""]
    lines.extend(f"{key} = {value}" for key, value in fields.items())
    lines.extend(
        [
            "",
            "BES remains in the mechanism suite and synthetic branch-heavy evaluation, but the timed mini-SWE real traces did not show an independent BES gain over ACD-only.",
            "The real mini-SWE main result should emphasize ACD/NISP context reuse and model-side savings, while reporting BES as not observed on this workload.",
        ]
    )
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fields


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/bes_positioning_pr3_v4.md")
    args = ap.parse_args()
    fields = write_bes_positioning(args.out)
    print("\n".join(f"{k} = {v}" for k, v in fields.items()))


if __name__ == "__main__":
    main()
