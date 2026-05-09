from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from agentweaver.simulator.wafer_config import WaferConfig

Coord = tuple[int, int]
Link = tuple[Coord, Coord]


def _norm_link(a: Coord, b: Coord) -> Link:
    return (a, b) if a <= b else (b, a)


@dataclass
class WaferMesh:
    config: WaferConfig
    link_traffic: dict[Link, float] = field(default_factory=lambda: defaultdict(float))
    background_traffic: dict[Link, float] = field(default_factory=lambda: defaultdict(float))

    def regions(self) -> list[Coord]:
        return [(r, c) for r in range(self.config.mesh_rows) for c in range(self.config.mesh_cols)]

    def manhattan(self, a: Coord, b: Coord) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def xy_path(self, a: Coord, b: Coord) -> list[Link]:
        links: list[Link] = []
        cur = a
        step = 1 if b[1] >= cur[1] else -1
        for c in range(cur[1], b[1], step):
            nxt = (cur[0], c + step)
            links.append(_norm_link(cur, nxt))
            cur = nxt
        step = 1 if b[0] >= cur[0] else -1
        for r in range(cur[0], b[0], step):
            nxt = (r + step, cur[1])
            links.append(_norm_link(cur, nxt))
            cur = nxt
        return links

    def path(self, a: Coord, b: Coord) -> list[Link]:
        return self.xy_path(a, b)

    def account_traffic(self, a: Coord, b: Coord, bytes_: float, background: bool = False) -> None:
        target = self.background_traffic if background else self.link_traffic
        for link in self.path(a, b):
            target[link] += bytes_

    def avg_hop_count(self) -> float:
        if not self.link_traffic:
            return 0.0
        total_bytes = sum(self.link_traffic.values())
        transfers = max(1, len(self.link_traffic))
        return total_bytes / transfers / max(1.0, total_bytes / max(1, transfers))

    def hotspot_ratio(self) -> float:
        if not self.link_traffic:
            return 1.0
        vals = list(self.link_traffic.values())
        avg = sum(vals) / len(vals)
        return max(vals) / avg if avg else 1.0

    def transfer_latency(self, hops: int, bytes_: float) -> float:
        bw_Bps = self.config.link_bandwidth_TBps * 1e12
        return hops * self.config.link_latency_ns * 1e-9 + bytes_ / bw_Bps

    def schedule_background_migration_if_slack_available(
        self, src: Coord, dst: Coord, bytes_: float, slack_threshold: float = 0.5
    ) -> bool:
        # Non-invasive migrations are admitted only when every path link is below
        # a simple utilization proxy. Replay uses this as a conservative slack test.
        links = self.path(src, dst)
        if not links:
            return True
        current_max = max(self.link_traffic.values() or [0.0])
        limit = max(1.0, current_max * slack_threshold)
        if all(self.link_traffic.get(l, 0.0) <= limit for l in links):
            self.account_traffic(src, dst, bytes_, background=True)
            return True
        return False


@dataclass
class GPUClusterSim:
    num_devices: int
    config: WaferConfig

    def transfer_latency(self, src: int, dst: int, bytes_: float) -> float:
        same_node = src // 8 == dst // 8
        bw = self.config.intra_node_bw_TBps if same_node else self.config.inter_node_bw_TBps
        lat = self.config.link_latency_ns if same_node else self.config.inter_node_latency_ns
        return lat * 1e-9 + bytes_ / (bw * 1e12)
