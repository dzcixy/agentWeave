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
from agentweaver.utils.io import ensure_dir, write_csv, write_json

ELIGIBLE_SHARED_TYPES = {"system", "tool_schema", "task", "repo", "history", "test_log"}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def branch_key(instance_id: str, branch_id: str) -> str:
    return f"{instance_id}:{branch_id}"


def weighted_median_bank(mesh: WaferMesh, consumer_regions: list[Coord]) -> Coord:
    if not consumer_regions:
        return (0, 0)
    return min(mesh.regions(), key=lambda r: (sum(mesh.manhattan(r, c) for c in consumer_regions), r))


def _hotspot_for_transfers(mesh: WaferMesh, transfers: list[tuple[Coord, Coord, float]]) -> float:
    traffic: dict[tuple[Coord, Coord], float] = defaultdict(float)
    for src, dst, bytes_ in transfers:
        for link in mesh.path(src, dst):
            traffic[link] += bytes_
    if not traffic:
        return 1.0
    vals = list(traffic.values())
    avg = sum(vals) / len(vals)
    return max(vals) / avg if avg else 1.0


def _compact_domains(mesh: WaferMesh, n: int) -> list[Coord]:
    center = (mesh.config.mesh_rows // 2, mesh.config.mesh_cols // 2)
    return sorted(mesh.regions(), key=lambda r: (mesh.manhattan(center, r), r))[:n]


class ACDMapper:
    def __init__(self, mesh: WaferMesh, enable_replication: bool = True, rho: float = 0.5):
        self.mesh = mesh
        self.enable_replication = enable_replication
        self.rho = rho

    def _extract(self, events: list[dict[str, Any]], segment_rows: list[dict[str, Any]]) -> tuple[
        list[str], dict[str, set[str]], dict[str, dict[str, Any]], dict[str, int]
    ]:
        branches = sorted(
            {branch_key(e["instance_id"], e["branch_id"]) for e in events if e.get("node_type") == "llm" and e["branch_id"] != "root"}
        )
        branch_segments: dict[str, set[str]] = defaultdict(set)
        segment_access: dict[str, int] = defaultdict(int)
        for e in events:
            if e.get("node_type") != "llm" or e["branch_id"] == "root":
                continue
            bk = branch_key(e["instance_id"], e["branch_id"])
            for s in e.get("context_segments", []):
                sid = s["segment_id"]
                branch_segments[bk].add(sid)
                segment_access[sid] += 1
        seg_info = {r["segment_id"]: r for r in segment_rows}
        return branches, branch_segments, seg_info, segment_access

    def _eligible(self, row: dict[str, Any]) -> bool:
        return (
            int(float(row.get("fanout", 0))) >= 2
            and str(row.get("exact_prefix_reusable", "True")).lower() == "true"
            and row.get("segment_type") in ELIGIBLE_SHARED_TYPES
        )

    def map(self, events: list[dict[str, Any]], segment_rows: list[dict[str, Any]]) -> dict[str, Any]:
        # Agent Context Domain:
        # A context domain is the set of branches/LLM nodes that consume the same
        # exact-prefix shared context segment. The goal is to co-place these
        # consumers and the shared KV segment in a compact mesh domain to reduce
        # average hop count and avoid hotspot traffic.
        branches, branch_segments, seg_info, segment_access = self._extract(events, segment_rows)
        eligible = [
            r
            for r in segment_rows
            if self._eligible(r)
        ]
        eligible.sort(
            key=lambda r: float(r.get("kv_bytes", 0))
            * max(1, int(float(r.get("access_count", segment_access.get(r["segment_id"], 1)))))
            * max(1, int(float(r.get("fanout", 1)))),
            reverse=True,
        )

        clusters: list[set[str]] = []
        assigned: set[str] = set()
        max_cluster = max(1, min(len(self.mesh.regions()), 8))
        for row in eligible:
            sid = row["segment_id"]
            consumers = {b for b, segs in branch_segments.items() if sid in segs and b not in assigned}
            if not consumers:
                continue
            consumers = set(sorted(consumers)[:max_cluster])
            clusters.append(consumers)
            assigned.update(consumers)
        for b in branches:
            if b not in assigned:
                clusters.append({b})

        domains = _compact_domains(self.mesh, len(branches))
        branch_to_region: dict[str, Coord] = {}
        cursor = 0
        for cluster in sorted(clusters, key=lambda c: (-len(c), sorted(c))):
            cluster_domains = domains[cursor : cursor + len(cluster)] or domains[: len(cluster)]
            for b, r in zip(sorted(cluster), cluster_domains):
                branch_to_region[b] = r
            cursor += len(cluster)
        for i, b in enumerate(branches):
            branch_to_region.setdefault(b, domains[i % len(domains)])

        bank_used: dict[Coord, float] = defaultdict(float)
        capacity = self.mesh.config.kv_capacity_bytes_per_die
        segment_to_bank: dict[str, Coord] = {}
        replicas: dict[str, list[Coord]] = {}
        segment_results: list[dict[str, Any]] = []
        before_transfers: list[tuple[Coord, Coord, float]] = []
        after_transfers: list[tuple[Coord, Coord, float]] = []
        naive_home = (0, 0)
        for sid, info in seg_info.items():
            kvb = float(info.get("kv_bytes", 0) or 0)
            access = max(1, int(float(info.get("access_count", segment_access.get(sid, 1)))))
            consumers = [b for b, segs in branch_segments.items() if sid in segs]
            cregions = [branch_to_region[b] for b in consumers if b in branch_to_region]
            if not cregions:
                continue
            eligible_shared = self._eligible(info)
            before_hops = [self.mesh.manhattan(naive_home, cr) for cr in cregions]
            before_bytes = sum(h * kvb * access for h in before_hops)
            for cr in cregions:
                before_transfers.append((naive_home, cr, kvb * access))

            if eligible_shared:
                home = weighted_median_bank(self.mesh, cregions)
            else:
                home = cregions[0]
            if bank_used[home] + kvb > capacity:
                home = min(self.mesh.regions(), key=lambda r: (bank_used[r] + kvb > capacity, self.mesh.manhattan(home, r)))
            segment_to_bank[sid] = home
            bank_used[home] += kvb

            replica_banks: list[Coord] = []
            homes = [home]
            if self.enable_replication and eligible_shared and len(set(cregions)) >= 4 and kvb > 0:
                far = max(set(cregions), key=lambda r: self.mesh.manhattan(home, r))
                without = sum(self.mesh.manhattan(home, cr) * kvb * access for cr in cregions)
                with_replica = sum(min(self.mesh.manhattan(home, cr), self.mesh.manhattan(far, cr)) * kvb * access for cr in cregions)
                copy_hops = max(1, self.mesh.manhattan(home, far))
                capacity_penalty = kvb if bank_used[far] + kvb > capacity else 0
                replication_cost = kvb * copy_hops + capacity_penalty
                if without - with_replica > self.rho * replication_cost and bank_used[far] + kvb <= capacity:
                    replica_banks.append(far)
                    replicas[sid] = [far]
                    homes.append(far)
                    bank_used[far] += kvb

            after_hops = [min(self.mesh.manhattan(h, cr) for h in homes) for cr in cregions]
            after_bytes = sum(h * kvb * access for h in after_hops)
            for cr in cregions:
                nearest = min(homes, key=lambda h: self.mesh.manhattan(h, cr))
                after_transfers.append((nearest, cr, kvb * access))
                self.mesh.account_traffic(nearest, cr, kvb * access)
            segment_results.append(
                {
                    "segment_id": sid,
                    "segment_type": info.get("segment_type", ""),
                    "fanout": len(set(consumers)),
                    "kv_bytes": kvb,
                    "home_bank": json.dumps(list(home)),
                    "replicas": json.dumps([list(r) for r in replica_banks]),
                    "avg_hops_before": sum(before_hops) / len(before_hops) if before_hops else 0.0,
                    "avg_hops_after": sum(after_hops) / len(after_hops) if after_hops else 0.0,
                    "noc_bytes_before": before_bytes,
                    "noc_bytes_after": after_bytes,
                    "hotspot_before": 1.0,
                    "hotspot_after": 1.0,
                    "eligible_shared": eligible_shared,
                }
            )
        hotspot_before = _hotspot_for_transfers(self.mesh, before_transfers)
        hotspot_after = _hotspot_for_transfers(self.mesh, after_transfers)
        for row in segment_results:
            row["hotspot_before"] = hotspot_before
            row["hotspot_after"] = hotspot_after
        total_before = sum(float(r["noc_bytes_before"]) for r in segment_results)
        total_after = sum(float(r["noc_bytes_after"]) for r in segment_results)
        avg_before = (
            sum(float(r["avg_hops_before"]) * int(r["fanout"]) for r in segment_results)
            / max(1, sum(int(r["fanout"]) for r in segment_results))
        )
        avg_after = (
            sum(float(r["avg_hops_after"]) * int(r["fanout"]) for r in segment_results)
            / max(1, sum(int(r["fanout"]) for r in segment_results))
        )
        return {
            "branch_to_region": {k: list(v) for k, v in branch_to_region.items()},
            "segment_to_bank": {k: list(v) for k, v in segment_to_bank.items()},
            "replicas": {k: [list(x) for x in v] for k, v in replicas.items()},
            "segments": segment_results,
            "avg_hops_before": avg_before,
            "avg_hops_after": avg_after,
            "noc_bytes_before": total_before,
            "noc_bytes_after": total_after,
            "hotspot_before": hotspot_before,
            "hotspot_after": hotspot_after,
            "estimated_noc_bytes": total_after,
            "avg_hops": avg_after,
            "hotspot_ratio": hotspot_after,
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
    write_csv(outp, result["segments"])
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
