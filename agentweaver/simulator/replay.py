from __future__ import annotations

import argparse
import heapq
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentweaver.analysis.context_segment_graph import kv_bytes_per_token
from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.acd_mapping import branch_key, run_mapping
from agentweaver.simulator.bes_scheduler import BESScheduler
from agentweaver.simulator.context_arena import ContextArena
from agentweaver.simulator.metrics import percentile
from agentweaver.simulator.nisp import BranchKVState, NISP, NISPDecision
from agentweaver.simulator.wafer_config import WaferConfig
from agentweaver.simulator.wafer_mesh import Coord, WaferMesh
from agentweaver.tracing.trace_schema import Event, Trace
from agentweaver.utils.io import read_json, write_csv


POLICIES = {
    "naive_wafer",
    "static_branch_pinning",
    "wafer_fcfs",
    "acd_only",
    "acd_bes",
    "acd_nisp",
    "full_agentweaver",
}


def _policy_flags(policy: str) -> dict[str, bool]:
    return {
        "acd": policy in {"acd_only", "acd_bes", "acd_nisp", "full_agentweaver"},
        "bes": policy in {"acd_bes", "full_agentweaver"},
        "nisp": policy in {"acd_nisp", "full_agentweaver"},
        "cancel": policy == "full_agentweaver",
        "static": policy == "static_branch_pinning",
        "release_on_tool": policy not in {"static_branch_pinning"},
    }


@dataclass(order=True)
class QueueEvent:
    time: float
    seq: int
    kind: str = field(compare=False)
    branch_id: str = field(compare=False)
    node: Event | None = field(default=None, compare=False)


@dataclass
class RunningLLM:
    node: Event
    region: Coord
    start: float
    duration: float
    cached_tokens: int
    delta_tokens: int
    fetch_latency: float


class EventDrivenReplay:
    def __init__(
        self,
        instance_id: str,
        scenario: str,
        events: list[Event],
        policy: str,
        cfg: WaferConfig,
        latency_model: LatencyModel,
        mapping: dict[str, Any] | None = None,
        run_id: str = "synthetic",
    ):
        self.instance_id = instance_id
        self.scenario = scenario
        self.events = sorted([e for e in events if e.branch_id != "root"], key=lambda e: (e.branch_id, e.step_id))
        self.policy = policy
        self.flags = _policy_flags(policy)
        self.cfg = cfg
        self.mesh = WaferMesh(cfg)
        self.latency_model = latency_model
        self.mapping = mapping or {}
        self.run_id = run_id
        self.bpt = kv_bytes_per_token()
        self.regions = self.mesh.regions()
        self.active_regions = self.regions
        self.free_regions: list[Coord] = list(self.active_regions)
        self.branch_region: dict[str, Coord] = {}
        self.arena = ContextArena(cfg.kv_capacity_bytes_per_die * cfg.num_regions)
        self.nisp = NISP(latency_model, mesh=self.mesh)
        self.scheduler = BESScheduler(latency_model, self.arena, free_regions=self.free_regions)
        self.ready_queue: deque[Event] = deque()
        self.eventq: list[QueueEvent] = []
        self.seq = 0
        self.running: dict[str, RunningLLM] = {}
        self.cancelled: set[str] = set()
        self.branch_done: set[str] = set()
        self.first_success_time: float | None = None
        self.accepted_branch = ""
        self.t0 = min((e.timestamp_ready or e.timestamp_start for e in self.events), default=0.0)
        self.by_branch: dict[str, list[Event]] = defaultdict(list)
        for ev in self.events:
            self.by_branch[ev.branch_id].append(ev)
        self.next_index: dict[str, int] = defaultdict(int)
        self.tool_samples: dict[str, list[float]] = defaultdict(list)
        for ev in self.events:
            if ev.node_type == "tool":
                self.tool_samples[ev.tool_type or "shell_other"].append(ev.latency)
        self.segment_to_bank = {k: tuple(v) for k, v in self.mapping.get("segment_to_bank", {}).items()}
        self.branch_to_region = {k: tuple(v) for k, v in self.mapping.get("branch_to_region", {}).items()}
        self.replicas = {k: [tuple(x) for x in v] for k, v in self.mapping.get("replicas", {}).items()}
        self.prefill_tokens = 0
        self.prefill_tokens_avoided = 0
        self.kv_hit_tokens = 0
        self.kv_miss_tokens = 0
        self.noc_bytes = 0.0
        self.hop_sum = 0.0
        self.hop_n = 0
        self.llm_busy_time = 0.0
        self.tool_blocked_region_time = 0.0
        self.blocked_compute_time_avoided = 0.0
        self.branch_wasted_tokens = 0
        self.safe_cancellation_savings = 0
        self.resume_prefill_tokens = 0
        self.region_last_free: dict[Coord, float] = {r: 0.0 for r in self.regions}

    def push(self, time: float, kind: str, branch_id: str, node: Event | None = None) -> None:
        self.seq += 1
        heapq.heappush(self.eventq, QueueEvent(time, self.seq, kind, branch_id, node))

    def _next_node(self, branch_id: str) -> Event | None:
        idx = self.next_index[branch_id]
        arr = self.by_branch[branch_id]
        if idx >= len(arr):
            return None
        node = arr[idx]
        self.next_index[branch_id] += 1
        return node

    def _peek_next_node(self, branch_id: str) -> Event | None:
        idx = self.next_index[branch_id]
        arr = self.by_branch[branch_id]
        return arr[idx] if idx < len(arr) else None

    def _preferred_region(self, branch_id: str) -> Coord:
        bk = branch_key(self.instance_id, branch_id)
        if bk in self.branch_to_region:
            return self.branch_to_region[bk]  # type: ignore[return-value]
        if branch_id in self.branch_region:
            return self.branch_region[branch_id]
        idx = sorted(self.by_branch).index(branch_id) % len(self.regions)
        return self.regions[idx]

    def _allocate_region(self, branch_id: str, now: float) -> Coord | None:
        if self.flags["static"] and branch_id in self.branch_region:
            return self.branch_region[branch_id]
        preferred = self._preferred_region(branch_id)
        if preferred in self.free_regions:
            self.free_regions.remove(preferred)
            region = preferred
        elif self.free_regions:
            region = self.free_regions.pop(0)
        else:
            return None
        self.branch_region[branch_id] = region
        idle = max(0.0, now - self.region_last_free.get(region, 0.0))
        self.scheduler.region_idle_time[region] = self.scheduler.region_idle_time.get(region, 0.0) + idle
        return region

    def _release_region(self, branch_id: str, region: Coord, now: float) -> None:
        if self.flags["static"]:
            return
        if region not in self.free_regions:
            self.free_regions.append(region)
        self.region_last_free[region] = now

    def _bank_for_segment(self, seg_id: str, region: Coord) -> Coord:
        homes = []
        if seg_id in self.segment_to_bank:
            homes.append(self.segment_to_bank[seg_id])
        homes.extend(self.replicas.get(seg_id, []))
        if not homes:
            return region
        return min(homes, key=lambda h: self.mesh.manhattan(h, region))

    def _shared_private_tokens(self, ev: Event) -> tuple[int, int]:
        shared = 0
        private = 0
        for seg in ev.context_segments:
            if seg.segment_type in {"system", "tool_schema", "task", "repo", "history", "test_log"}:
                shared += seg.length
            elif seg.segment_type in {"branch_suffix", "patch", "scratchpad", "observation"}:
                private += seg.length
        if private == 0:
            private = max(1, ev.input_tokens - shared)
        return shared, private

    def _nisp_restore_cached(self, ev: Event) -> tuple[int, int]:
        decision = self.nisp.restore_state(ev.branch_id) if self.flags["nisp"] and ev.step_id >= 3 else None
        if decision is None:
            cached = self.arena.match(ev.context_segments) if self.flags["acd"] else 0
            return cached, max(0, ev.input_tokens - cached)
        cached = min(ev.input_tokens, decision.cached_tokens)
        resume = min(ev.input_tokens, decision.resume_prefill_tokens)
        return max(cached, ev.input_tokens - resume), resume

    def _start_llm(self, ev: Event, now: float, region: Coord) -> None:
        cached, delta = self._nisp_restore_cached(ev)
        if self.flags["acd"] and cached == 0:
            cached = self.arena.match(ev.context_segments)
            delta = max(0, ev.input_tokens - cached)
        fetch_latency = 0.0
        for seg in ev.context_segments:
            if self.flags["acd"] and seg.segment_id in self.arena.resident:
                bank = self._bank_for_segment(seg.segment_id, region)
                hops = self.mesh.manhattan(bank, region)
                bytes_ = seg.kv_bytes or seg.length * self.bpt
                self.mesh.account_traffic(bank, region, bytes_)
                self.noc_bytes += bytes_
                self.hop_sum += hops
                self.hop_n += 1
                fetch_latency += self.mesh.transfer_latency(hops, bytes_)
        duration = self.latency_model.predict_llm(ev.input_tokens, ev.output_tokens, cached) + fetch_latency
        self.prefill_tokens += delta
        self.prefill_tokens_avoided += cached
        self.kv_hit_tokens += cached
        self.kv_miss_tokens += delta
        if ev.step_id >= 3:
            self.resume_prefill_tokens += delta
        self.running[ev.branch_id] = RunningLLM(ev, region, now, duration, cached, delta, fetch_latency)
        self.push(now + duration, "LLM_DONE", ev.branch_id, ev)

    def _try_schedule(self, now: float) -> None:
        while self.ready_queue and self.free_regions:
            ready = [ev for ev in self.ready_queue if ev.branch_id not in self.cancelled]
            if not ready:
                self.ready_queue.clear()
                return
            if self.flags["bes"]:
                selected = max(ready, key=lambda ev: self.scheduler.score(ev, now, avg_hops=max(1.0, self.hop_sum / max(1, self.hop_n))))
            else:
                selected = ready[0]
            self.ready_queue.remove(selected)
            region = self._allocate_region(selected.branch_id, now)
            if region is None:
                self.ready_queue.appendleft(selected)
                return
            self.push(now, "LLM_START", selected.branch_id, selected)

    def _cancel_siblings(self, winner: str, now: float) -> None:
        for branch_id, nodes in self.by_branch.items():
            if branch_id == winner or branch_id in self.branch_done:
                continue
            self.cancelled.add(branch_id)
            for ev in nodes[self.next_index[branch_id] :]:
                if ev.node_type == "llm":
                    self.safe_cancellation_savings += ev.input_tokens + ev.output_tokens
            if branch_id in self.running:
                run = self.running.pop(branch_id)
                self.safe_cancellation_savings += run.node.input_tokens + run.node.output_tokens
                self._release_region(branch_id, run.region, now)
            self.push(now, "BRANCH_CANCEL", branch_id, None)

    def run(self) -> dict[str, Any]:
        for branch_id in sorted(self.by_branch):
            node = self._next_node(branch_id)
            if node and node.node_type == "llm":
                self.push(max(0.0, node.timestamp_ready - self.t0), "LLM_READY", branch_id, node)
        last_time = 0.0
        while self.eventq:
            qe = heapq.heappop(self.eventq)
            now = qe.time
            last_time = max(last_time, now)
            if qe.branch_id in self.cancelled and qe.kind not in {"BRANCH_CANCEL"}:
                continue
            if qe.kind == "LLM_READY":
                self.ready_queue.append(qe.node)  # type: ignore[arg-type]
                self._try_schedule(now)
            elif qe.kind == "LLM_START":
                self._start_llm(qe.node, now, self.branch_region[qe.branch_id])  # type: ignore[arg-type]
            elif qe.kind == "LLM_DONE":
                run = self.running.pop(qe.branch_id)
                self.llm_busy_time += run.duration
                self.scheduler.region_busy_time[run.region] = self.scheduler.region_busy_time.get(run.region, 0.0) + run.duration
                self.arena.now = now
                self.arena.insert(run.node.context_segments, self.bpt)
                next_node = self._next_node(qe.branch_id)
                if next_node and next_node.node_type == "tool":
                    self.push(now, "TOOL_START", qe.branch_id, next_node)
                elif next_node and next_node.node_type == "verifier":
                    if self.flags["release_on_tool"]:
                        self._release_region(qe.branch_id, run.region, now)
                    self.push(now, "VERIFY_START", qe.branch_id, next_node)
                self._try_schedule(now)
            elif qe.kind == "TOOL_START":
                node = qe.node
                region = self.branch_region.get(qe.branch_id)
                if region is not None and self.flags["release_on_tool"]:
                    self._release_region(qe.branch_id, region, now)
                    self.blocked_compute_time_avoided += node.latency if node else 0.0
                else:
                    self.tool_blocked_region_time += node.latency if node else 0.0
                if self.flags["nisp"] and node:
                    next_llm = self._peek_next_node(qe.branch_id)
                    if next_llm:
                        shared, private = self._shared_private_tokens(next_llm)
                        state = BranchKVState(
                            branch_id=qe.branch_id,
                            shared_prefix_tokens=shared,
                            private_suffix_tokens=private,
                            observation_tokens=node.observation_tokens or 0,
                            full_context_tokens=next_llm.input_tokens,
                            shared_prefix_kv_bytes=shared * self.bpt,
                            private_suffix_kv_bytes=private * self.bpt,
                            region=region,
                        )
                        self.nisp.park_branch(
                            state,
                            self.tool_samples.get(node.tool_type or "shell_other", []),
                            self.bpt,
                            self.arena.occupancy(),
                            self.arena.capacity_bytes,
                            src=region,
                            dst=region,
                        )
                self.push(now + (node.latency if node else 0.0), "TOOL_DONE", qe.branch_id, node)
                self._try_schedule(now)
            elif qe.kind == "TOOL_DONE":
                next_node = self._next_node(qe.branch_id)
                if next_node and next_node.node_type == "llm":
                    self.push(now, "LLM_READY", qe.branch_id, next_node)
                self._try_schedule(now)
            elif qe.kind == "VERIFY_START":
                self.push(now + (qe.node.latency if qe.node else 0.0), "VERIFY_DONE", qe.branch_id, qe.node)
            elif qe.kind == "VERIFY_DONE":
                node = qe.node
                success = bool(node and (node.success or node.verifier_result == "pass"))
                self.branch_done.add(qe.branch_id)
                if success and self.first_success_time is None:
                    self.first_success_time = now
                    self.accepted_branch = qe.branch_id
                    if self.flags["cancel"]:
                        self._cancel_siblings(qe.branch_id, now)
                region = self.branch_region.get(qe.branch_id)
                if self.flags["static"] and region is not None:
                    self._release_region(qe.branch_id, region, now)
                self._try_schedule(now)
            elif qe.kind == "BRANCH_CANCEL":
                self.branch_done.add(qe.branch_id)
        jct = last_time
        if self.first_success_time is not None:
            for branch_id, nodes in self.by_branch.items():
                if branch_id == self.accepted_branch:
                    continue
                for ev in nodes:
                    if ev.node_type == "llm" and ev.timestamp_start - self.t0 <= jct:
                        self.branch_wasted_tokens += ev.input_tokens + ev.output_tokens
        nisp_metrics = self.nisp.metrics()
        total_region_time = self.llm_busy_time + self.tool_blocked_region_time + sum(self.scheduler.region_idle_time.values())
        return {
            "run_id": self.run_id,
            "scenario": self.scenario,
            "instance_id": self.instance_id,
            "policy": self.policy,
            "mesh_rows": self.cfg.mesh_rows,
            "mesh_cols": self.cfg.mesh_cols,
            "branch_fanout": len(self.by_branch),
            "jct": jct,
            "time_to_first_success": self.first_success_time if self.first_success_time is not None else jct,
            "success_any": self.first_success_time is not None,
            "prefill_tokens": self.prefill_tokens,
            "prefill_tokens_avoided": self.prefill_tokens_avoided,
            "kv_hit_tokens": self.kv_hit_tokens,
            "kv_miss_tokens": self.kv_miss_tokens,
            "noc_bytes": self.noc_bytes,
            "avg_hops": self.hop_sum / self.hop_n if self.hop_n else 0.0,
            "hotspot_ratio": self.mesh.hotspot_ratio(),
            "region_utilization": self.llm_busy_time / max(1e-9, total_region_time),
            "tool_blocked_region_time": self.tool_blocked_region_time,
            "blocked_compute_time_avoided": self.blocked_compute_time_avoided,
            "state_migration_exposed_latency": nisp_metrics["state_migration_exposed_latency"],
            "branch_wasted_tokens": max(0, self.branch_wasted_tokens - self.safe_cancellation_savings),
            "safe_cancellation_savings": self.safe_cancellation_savings,
            "hot_count": nisp_metrics["hot_count"],
            "warm_count": nisp_metrics["warm_count"],
            "cold_count": nisp_metrics["cold_count"],
            "resume_prefill_tokens": self.resume_prefill_tokens if not self.flags["nisp"] else nisp_metrics["resume_prefill_tokens"],
        }


def _load_mapping(processed: Path, cfg_path: str | Path, policy: str) -> dict[str, Any]:
    if not _policy_flags(policy)["acd"]:
        return {}
    mp = processed / f"acd_mapping_{Path(cfg_path).stem}.json"
    if not mp.exists():
        run_mapping(processed, cfg_path, processed / f"acd_mapping_{Path(cfg_path).stem}.csv")
    return read_json(mp)


def _group_by_instance(events: list[Event]) -> dict[str, list[Event]]:
    grouped: dict[str, list[Event]] = defaultdict(list)
    for ev in events:
        grouped[ev.instance_id].append(ev)
    return grouped


def replay(processed: str | Path, wafer_config: str | Path, policy: str, out: str | Path, run_id: str = "synthetic") -> list[dict[str, Any]]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy}; expected {sorted(POLICIES)}")
    processed = Path(processed)
    cfg = WaferConfig.from_yaml(wafer_config)
    trace = Trace.from_jsonl(processed / "events.jsonl")
    mapping = _load_mapping(processed, wafer_config, policy)
    lm = LatencyModel.load(processed / "h100_latency_model.json")
    rows = [
        EventDrivenReplay(instance, instance, evs, policy, cfg, lm, mapping, run_id=run_id).run()
        for instance, evs in sorted(_group_by_instance(trace.events).items())
    ]
    if rows:
        rows.append(
            {
                "run_id": run_id,
                "scenario": "AGGREGATE",
                "instance_id": "AGGREGATE",
                "policy": policy,
                "mesh_rows": cfg.mesh_rows,
                "mesh_cols": cfg.mesh_cols,
                "branch_fanout": sum(float(r["branch_fanout"]) for r in rows) / len(rows),
                "jct": sum(float(r["jct"]) for r in rows) / len(rows),
                "time_to_first_success": sum(float(r["time_to_first_success"]) for r in rows) / len(rows),
                "success_any": any(bool(r["success_any"]) for r in rows),
                "prefill_tokens": sum(int(r["prefill_tokens"]) for r in rows),
                "prefill_tokens_avoided": sum(int(r["prefill_tokens_avoided"]) for r in rows),
                "kv_hit_tokens": sum(int(r["kv_hit_tokens"]) for r in rows),
                "kv_miss_tokens": sum(int(r["kv_miss_tokens"]) for r in rows),
                "noc_bytes": sum(float(r["noc_bytes"]) for r in rows),
                "avg_hops": sum(float(r["avg_hops"]) for r in rows) / len(rows),
                "hotspot_ratio": max(float(r["hotspot_ratio"]) for r in rows),
                "region_utilization": sum(float(r["region_utilization"]) for r in rows) / len(rows),
                "tool_blocked_region_time": sum(float(r["tool_blocked_region_time"]) for r in rows),
                "blocked_compute_time_avoided": sum(float(r["blocked_compute_time_avoided"]) for r in rows),
                "state_migration_exposed_latency": sum(float(r["state_migration_exposed_latency"]) for r in rows),
                "branch_wasted_tokens": sum(int(r["branch_wasted_tokens"]) for r in rows),
                "safe_cancellation_savings": sum(int(r["safe_cancellation_savings"]) for r in rows),
                "hot_count": sum(int(r["hot_count"]) for r in rows),
                "warm_count": sum(int(r["warm_count"]) for r in rows),
                "cold_count": sum(int(r["cold_count"]) for r in rows),
                "resume_prefill_tokens": sum(int(r["resume_prefill_tokens"]) for r in rows),
                "p50": percentile([float(r["jct"]) for r in rows], 50),
                "p95": percentile([float(r["jct"]) for r in rows], 95),
                "p99": percentile([float(r["jct"]) for r in rows], 99),
            }
        )
    write_csv(out, rows)
    if "real_agentlike" in str(processed) or "real_agentlike" in str(out):
        write_csv("data/results/real_agentlike_replay_summary.csv", rows)
        _update_pr2_real_agentlike_report(str(processed), str(out))
    return rows


def _update_pr2_real_agentlike_report(processed: str, out: str) -> None:
    report = Path("data/results/pr2_h100_profile_report.md")
    fields: dict[str, str] = {}
    if report.exists():
        for line in report.read_text(encoding="utf-8").splitlines():
            if " = " in line:
                k, v = line.split(" = ", 1)
                fields[k.strip()] = v.strip()
    defaults = {
        "PR1_GATE": "PASS",
        "H100_PROFILE": fields.get("H100_PROFILE", "NOT_RUN"),
        "LATENCY_MODEL_QUALITY": fields.get("LATENCY_MODEL_QUALITY", "NOT_RUN"),
        "REAL_AGENTLIKE_TRACE": "PASS",
        "VLLM_SERVER_URL": fields.get("VLLM_SERVER_URL", ""),
        "MODEL_PATH": fields.get("MODEL_PATH", ""),
        "MODEL_NAME": fields.get("MODEL_NAME", ""),
        "TOKENIZER_PATH": fields.get("TOKENIZER_PATH", ""),
        "VLLM_METRICS_URL": fields.get("VLLM_METRICS_URL", ""),
        "VLLM_METRICS_AVAILABLE": fields.get("VLLM_METRICS_AVAILABLE", "false"),
        "PREFIX_CACHE_METRICS_OBSERVED": fields.get("PREFIX_CACHE_METRICS_OBSERVED", "false"),
        "PROMPT_FACTORY_512": fields.get("PROMPT_FACTORY_512", ""),
        "PROMPT_FACTORY_4096": fields.get("PROMPT_FACTORY_4096", ""),
        "PROMPT_FACTORY_8192": fields.get("PROMPT_FACTORY_8192", ""),
        "SHARED_PREFIX_EXACT_MATCH": fields.get("SHARED_PREFIX_EXACT_MATCH", ""),
        "LENGTH_SWEEP_SUCCESS_CASES": fields.get("LENGTH_SWEEP_SUCCESS_CASES", "0"),
        "LENGTH_SWEEP_FAILED_CASES": fields.get("LENGTH_SWEEP_FAILED_CASES", "0"),
        "CONCURRENCY_SWEEP_SUCCESS_CASES": fields.get("CONCURRENCY_SWEEP_SUCCESS_CASES", "0"),
        "CONCURRENCY_SWEEP_FAILED_CASES": fields.get("CONCURRENCY_SWEEP_FAILED_CASES", "0"),
        "PREFIX_SWEEP_SUCCESS_CASES": fields.get("PREFIX_SWEEP_SUCCESS_CASES", "0"),
        "PREFIX_SWEEP_FAILED_CASES": fields.get("PREFIX_SWEEP_FAILED_CASES", "0"),
        "MAX_SUCCESSFUL_CONCURRENCY": fields.get("MAX_SUCCESSFUL_CONCURRENCY", "0"),
        "MEDIAN_LATENCY_MODEL_ERROR": fields.get("MEDIAN_LATENCY_MODEL_ERROR", ""),
        "P95_LATENCY_MODEL_ERROR": fields.get("P95_LATENCY_MODEL_ERROR", ""),
        "PROFILE_OUTPUT_PATHS": fields.get("PROFILE_OUTPUT_PATHS", ""),
        "REAL_AGENTLIKE_TRACE_PATH": processed.replace("data/processed/", "data/traces/"),
        "REAL_AGENTLIKE_REPLAY_PATH": out,
        "REMAINING_BLOCKERS_FOR_SWE_AGENT": fields.get(
            "REMAINING_BLOCKERS_FOR_SWE_AGENT", "collect real SWE-agent trajectories and official harness correctness"
        ),
    }
    body = "# PR2 H100 Profile Report\n\n" + "\n".join(f"{k} = {v}" for k, v in defaults.items()) + "\n"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(body, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--wafer-config", default="configs/wafer_6x6.yaml")
    ap.add_argument("--policy", default="full_agentweaver", choices=sorted(POLICIES))
    ap.add_argument("--out", default="data/results/wafer_replay_summary.csv")
    ap.add_argument("--run-id", default="synthetic")
    args = ap.parse_args()
    rows = replay(args.processed, args.wafer_config, args.policy, args.out, run_id=args.run_id)
    print(json.dumps(rows[-1] if rows else {}, indent=2))


if __name__ == "__main__":
    main()
