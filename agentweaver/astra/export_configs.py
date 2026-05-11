from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentweaver.utils.io import ensure_dir


def write_smoke_configs(out_dir: str | Path = "data/astra_configs/smoke", *, mesh_rows: int = 4, mesh_cols: int = 4) -> dict[str, Path]:
    out = ensure_dir(out_dir)
    configs: dict[str, dict[str, Any]] = {
        "system.json": {
            "format": "agentweaver_astra_intermediate_config",
            "mesh_rows": mesh_rows,
            "mesh_cols": mesh_cols,
            "num_npus": mesh_rows * mesh_cols,
            "clock_hz": 1_000_000_000,
        },
        "network.json": {
            "format": "agentweaver_astra_intermediate_config",
            "topology": "2d_mesh",
            "link_bandwidth_bytes_per_sec": 200_000_000_000,
            "routing": "dimension_order",
        },
        "memory.json": {
            "format": "agentweaver_astra_intermediate_config",
            "hbm_capacity_bytes_per_npu": 80 * 1024**3,
            "sram_capacity_bytes_per_npu": 32 * 1024**3,
            "kv_bytes_per_token": 524_288,
        },
    }
    written: dict[str, Path] = {}
    for name, obj in configs.items():
        path = out / name
        path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written[name] = path
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/astra_configs/smoke")
    ap.add_argument("--mesh-rows", type=int, default=4)
    ap.add_argument("--mesh-cols", type=int, default=4)
    args = ap.parse_args()
    written = write_smoke_configs(args.out_dir, mesh_rows=args.mesh_rows, mesh_cols=args.mesh_cols)
    print(json.dumps({k: str(v) for k, v in written.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

