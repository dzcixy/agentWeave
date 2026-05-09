from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentweaver.simulator.wafer_config import WaferConfig
from agentweaver.simulator.wafer_mesh import Coord, WaferMesh
from agentweaver.tracing.trace_schema import Trace
from agentweaver.utils.io import ensure_dir, read_json, write_csv, write_json


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def weighted_median_bank(mesh: WaferMesh, consumer_regions: list[Coord]) -> Coord:
    if not consumer_regions:
        return (0, 0)
    best = None
    best_cost = math.inf
    for r in mesh.regions():
        cost = sum(mesh.manhattan(r, c) for c in consumer_regions)
        if cost < best_cost:
            best, best_cost = r, cost
    return best or (0, 0)


class ACDMapper:
    def __init__(self, mesh: WaferMesh, enable_replication: bool = True, rho: float = 1.2):
        self.mesh = mesh
        self.enable_replication = enable_replication
        self.rho = rho

    def map(self, events: list[dict[str, Any]], segment_rows: list[dict[str, Any]]) -> dict[str, Any]:
        # Agent Context Domain:
        # A context domain is the set of branches/LLM nodes that consume the same
        # exact-prefix shared context segment. The goal is to co-place these
        # consumers and the shared KV segment in a compact mesh domain to reduce
        # average hop count and avoid hotspot traffic.
        branches = sorted({e["branch_id"] for e in events if e.get("node_type") == "llm"})
        branch_segments: dict[str, set[str]] = defaultdict(set)
        node_branch: dict[str, str] = {}
        for e in events:
            if e.get("node_type") != "llm":
                continue
            node_branch[e["node_id"]] = e["branch_id"]
            for s in e.get("context_segments", []):
                branch_segments[e["branch_id"]].add(s["segment_id"])
        seg_info = {r["segment_id"]: r for r in segment_rows}
        high_weight = sorted(
            [
                r
                for r in segment_rows
                if int(float(r.get("fanout", 0))) > 1 and str(r.get("exact_prefix_reusable", "True")) == "True"
            ],
            key=lambda r: float(r.get("kv_bytes", 0)) * float(r.get("access_count", 1)),
            reverse=True,
        )
        branch_score: dict[str, float] = defaultdict(float)
        for r in high_weight:
            sid = r["segment_id"]
            weight = float(r.get("kv_bytes", 0)) * float(r.get("access_count", 1))
            for b, segs in branch_segments.items():
                if sid in segs:
                    branch_score[b] += weight
        ordered = sorted(branches, key=lambda b: (-branch_score[b], b))
        regions = self.mesh.regions()
        branch_to_region: dict[str, Coord] = {}
        for idx, branch in enumerate(ordered):
            if idx == 0:
                branch_to_region[branch] = regions[len(regions) // 2]
            else:
                used = list(branch_to_region.values())
                candidates = sorted(regions, key=lambda r: (sum(self.mesh.manhattan(r, u) for u in used), r))
                branch_to_region[branch] = next(r for r in candidates if r not in used)
        bank_used: dict[Coord, float] = defaultdict(float)
        capacity = self.mesh.config.kv_capacity_bytes_per_die
        segment_to_bank: dict[str, Coord] = {}
        replicas: dict[str, list[Coord]] = {}
        noc_bytes = 0.0
        hop_weight = 0.0
        access_weight = 0.0
        for sid, info in seg_info.items():
            kvb = float(info.get("kv_bytes", 0) or 0)
            consumers = [b for b, segs in branch_segments.items() if sid in segs]
            cregions = [branch_to_region[b] for b in consumers if b in branch_to_region]
            home = weighted_median_bank(self.mesh, cregions)
            if bank_used[home] + kvb > capacity:
                home = min(self.mesh.regions(), key=lambda r: (bank_used[r] + kvb > capacity, self.mesh.manhattan(home, r)))
            segment_to_bank[sid] = home
            bank_used[home] += kvb
            for cr in cregions:
                hops = self.mesh.manhattan(home, cr)
                noc_bytes += hops * kvb
                hop_weight += hops
                access_weight += 1
                self.mesh.account_traffic(home, cr, kvb)
            if self.enable_replication and len(cregions) >= 4 and kvb > 0:
                far = max(cregions, key=lambda r: self.mesh.manhattan(home, r))
                saved = self.mesh.manhattan(home, far) * kvb
                cost = kvb + self.mesh.manhattan(home, far) * kvb
                if saved > self.rho * cost and bank_used[far] + kvb <= capacity:
                    replicas.setdefault(sid, []).append(far)
                    bank_used[far] += kvb
        avg_hops = hop_weight / access_weight if access_weight else 0.0
        return {
            "branch_to_region": {k: list(v) for k, v in branch_to_region.items()},
            "segment_to_bank": {k: list(v) for k, v in segment_to_bank.items()},
            "replicas": {k: [list(x) for x in v] for k, v in replicas.items()},
            "estimated_noc_bytes": noc_bytes,
            "avg_hops": avg_hops,
            "hotspot_ratio": self.mesh.hotspot_ratio(),
            "kv_capacity_usage": sum(bank_used.values()) / max(1, capacity * len(self.mesh.regions())),
        }


def run_mapping(processed: str | Path, config: str | Path, out: str | Path) -> dict[str, Any]:
    processed = Path(processed)
    cfg = WaferConfig.from_yaml(config)
    mesh = WaferMesh(cfg)
    trace = Trace.from_jsonl(processed / "events.jsonl")
    events = [e.__dict__ | {"context_segments": [s.__dict__ for s in e.context_segments]} for e in trace.events]
    rows = _load_rows(processed / "context_segments.csv")
    result = ACDMapper(mesh, enable_replication=cfg.enable_replication).map(events, rows)
    outp = Path(out)
    ensure_dir(outp.parent)
    write_json(outp.with_suffix(".json"), result)
    write_csv(
        outp,
        [
            {
                "policy": "acd",
                "mesh": f"{cfg.mesh_rows}x{cfg.mesh_cols}",
                "avg_hops": result["avg_hops"],
                "noc_bytes": result["estimated_noc_bytes"],
                "hotspot_ratio": result["hotspot_ratio"],
                "kv_capacity_usage": result["kv_capacity_usage"],
                "replicas": sum(len(v) for v in result["replicas"].values()),
            }
        ],
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", "--processed", dest="processed", required=True)
    ap.add_argument("--config", default="configs/wafer_6x6.yaml")
    ap.add_argument("--out", default="data/results/acd_mapping.csv")
    args = ap.parse_args()
    print(json.dumps(run_mapping(args.processed, args.config, args.out), indent=2)[:4000])


if __name__ == "__main__":
    main()
