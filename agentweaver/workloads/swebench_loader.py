from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from agentweaver.workloads.synthetic_fork_join import generate_from_config
from agentweaver.utils.io import read_yaml, write_json

LOG = logging.getLogger(__name__)


def load_swebench_lite(config: str | Path, out: str | Path | None = None) -> list[dict[str, Any]]:
    cfg = read_yaml(config)
    dataset_name = cfg.get("dataset", "princeton-nlp/SWE-bench_Lite")
    n = int(cfg.get("num_instances", 10))
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(dataset_name, split="test")
        rows = [dict(ds[i]) for i in range(min(n, len(ds)))]
    except Exception as exc:
        if not cfg.get("synthetic_fallback", True):
            raise RuntimeError(f"failed to load {dataset_name}: {exc}") from exc
        LOG.warning("SWE-bench unavailable, using synthetic fallback: %s", exc)
        generate_from_config("configs/small_sanity.yaml", "data/traces", "swebench_fallback")
        rows = [{"instance_id": f"synthetic_{i:04d}", "fallback": True} for i in range(n)]
    if out:
        write_json(out, rows)
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/swebench_lite.yaml")
    ap.add_argument("--out", default="data/processed/swebench_instances.json")
    args = ap.parse_args()
    print(json.dumps({"instances": len(load_swebench_lite(args.config, args.out)), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
