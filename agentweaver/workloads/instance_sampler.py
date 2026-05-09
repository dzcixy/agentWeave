from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from agentweaver.utils.io import read_json, write_json


def sample_instances(path: str | Path, n: int, seed: int = 1, out: str | Path | None = None) -> list[dict[str, Any]]:
    rows = read_json(path)
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    rows = rows[:n]
    if out:
        write_json(out, rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data/processed/sample_instances.json")
    args = ap.parse_args()
    print(json.dumps({"sampled": len(sample_instances(args.instances, args.n, args.seed, args.out)), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
