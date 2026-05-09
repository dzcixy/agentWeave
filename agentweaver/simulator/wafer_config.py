from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentweaver.utils.io import read_yaml


@dataclass
class WaferConfig:
    mesh_rows: int = 6
    mesh_cols: int = 6
    die_compute_scale: float = 1.0
    die_memory_capacity_gb: float = 80.0
    kv_budget_ratio: float = 0.3
    link_bandwidth_TBps: float = 1.0
    link_latency_ns: float = 500.0
    routing: str = "xy"
    region_granularity: str = "die"
    enable_replication: bool = True
    enable_noc_slack: bool = True
    intra_node_bw_TBps: float = 0.9
    inter_node_bw_TBps: float = 0.1
    inter_node_latency_ns: float = 3000.0

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "WaferConfig":
        if path is None:
            return cls()
        data = read_yaml(path)
        if "wafer" in data:
            data = data["wafer"]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def num_regions(self) -> int:
        return self.mesh_rows * self.mesh_cols

    @property
    def kv_capacity_bytes_per_die(self) -> int:
        return int(self.die_memory_capacity_gb * (1024**3) * self.kv_budget_ratio)
