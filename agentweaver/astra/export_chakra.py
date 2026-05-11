from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentweaver.analysis.context_segment_graph import kv_bytes_per_token
from agentweaver.astra.chakra_node import ChakraNode
from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.tracing.trace_schema import Event, Trace
from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.io import ensure_dir


CLOCK_HZ = 1_000_000_000


def _cycles(seconds: float) -> int:
    return max(0, int(round(max(0.0, seconds) * CLOCK_HZ)))


def _npu_for_event(ev: Event, npu_count: int) -> int:
    key = f"{ev.instance_id}:{ev.branch_id}:{ev.shared_prefix_id or ev.prompt_hash or ev.node_id}"
    return int(stable_hash(key), 16) % max(1, npu_count)


class ChakraExporter:
    def __init__(self, latency_model: LatencyModel | None = None, *, npu_count: int = 16) -> None:
        self.lm = latency_model or LatencyModel()
        self.npu_count = max(1, npu_count)
        self.nodes: list[ChakraNode] = []
        self.next_id = 1
        self.last_by_branch: dict[str, int] = {}
        self.stats = {
            "compute_nodes": 0,
            "communication_nodes": 0,
            "memory_nodes": 0,
            "delay_nodes": 0,
            "dependency_count": 0,
            "estimated_communication_bytes": 0,
            "estimated_compute_time": 0.0,
            "local_memory_bytes": 0,
            "remote_communication_bytes": 0,
        }

    def _add(self, name: str, typ: str, npu: int, *, seconds: float = 0.0, size_bytes: int = 0, deps: list[int] | None = None, metadata: dict[str, Any] | None = None) -> int:
        node = ChakraNode(
            id=self.next_id,
            name=name,
            type=typ,  # type: ignore[arg-type]
            npu_id=npu,
            duration_cycles=_cycles(seconds),
            size_bytes=int(max(0, size_bytes)),
            deps=list(dict.fromkeys(deps or [])),
            metadata=metadata or {},
        )
        self.next_id += 1
        self.nodes.append(node)
        self.stats[f"{typ}_nodes"] += 1
        self.stats["dependency_count"] += len(node.deps)
        if typ == "communication":
            self.stats["estimated_communication_bytes"] += node.size_bytes
        if typ == "compute":
            self.stats["estimated_compute_time"] += seconds
        return node.id

    def _deps_for(self, ev: Event) -> list[int]:
        dep = self.last_by_branch.get(ev.branch_id)
        return [dep] if dep is not None else []

    def _finish_event(self, ev: Event, node_id: int) -> None:
        self.last_by_branch[ev.branch_id] = node_id

    def add_event(self, ev: Event) -> None:
        npu = _npu_for_event(ev, self.npu_count)
        deps = self._deps_for(ev)
        if ev.node_type == "llm":
            kv_bytes = int(sum(ref.length for ref in ev.context_segments) * kv_bytes_per_token())
            mem_id = self._add(
                f"{ev.node_id}:context_fetch",
                "memory",
                npu,
                seconds=kv_bytes / 1e12,
                size_bytes=kv_bytes,
                deps=deps,
                metadata={"event_id": ev.event_id, "mapping": "KV/context fetch"},
            )
            comm_id = self._add(
                f"{ev.node_id}:remote_kv",
                "communication",
                npu,
                seconds=kv_bytes / 2e11,
                size_bytes=kv_bytes,
                deps=[mem_id],
                metadata={"event_id": ev.event_id, "mapping": "remote KV/context movement"},
            )
            cached = int(ev.kv_cache_hit_tokens or 0)
            prefill = self.lm.predict_prefill(max(0, ev.input_tokens - cached))
            decode = self.lm.predict_decode(ev.input_tokens, ev.output_tokens)
            prefill_id = self._add(
                f"{ev.node_id}:prefill",
                "compute",
                npu,
                seconds=prefill,
                deps=[comm_id],
                metadata={"event_id": ev.event_id, "mapping": "LLM prefill", "input_tokens": ev.input_tokens, "cached_tokens": cached},
            )
            decode_id = self._add(
                f"{ev.node_id}:decode",
                "compute",
                npu,
                seconds=decode,
                deps=[prefill_id],
                metadata={"event_id": ev.event_id, "mapping": "LLM decode", "output_tokens": ev.output_tokens},
            )
            self._finish_event(ev, decode_id)
        elif ev.node_type == "tool":
            seconds = float(ev.tool_latency if ev.tool_latency is not None else ev.latency or 0.0)
            node_id = self._add(
                f"{ev.node_id}:tool_delay",
                "delay",
                npu,
                seconds=seconds,
                deps=deps,
                metadata={"event_id": ev.event_id, "mapping": "tool call", "tool_type": ev.tool_type, "command": ev.command},
            )
            self._finish_event(ev, node_id)
        elif ev.node_type == "verifier":
            seconds = float(ev.latency or 0.0)
            node_id = self._add(
                f"{ev.node_id}:verifier",
                "delay" if seconds else "compute",
                npu,
                seconds=seconds,
                deps=deps,
                metadata={"event_id": ev.event_id, "mapping": "verifier", "verifier_result": ev.verifier_result},
            )
            self._finish_event(ev, node_id)
        else:
            node_id = self._add(
                f"{ev.node_id}:{ev.node_type}",
                "compute",
                npu,
                seconds=float(ev.latency or 0.0),
                deps=deps,
                metadata={"event_id": ev.event_id, "mapping": ev.node_type},
            )
            self._finish_event(ev, node_id)

    def add_event_policy_aware(self, ev: Event, schedule: dict[str, Any], policy: str) -> None:
        npu = int(schedule.get("region_id", _npu_for_event(ev, self.npu_count))) % self.npu_count
        deps = self._deps_for(ev)
        if ev.node_type != "llm":
            self.add_event(ev)
            return
        total_context_tokens = sum(ref.length for ref in ev.context_segments)
        cached_tokens = min(int(schedule.get("cached_tokens", 0)), ev.input_tokens, total_context_tokens)
        local_bytes = int(schedule.get("local_context_bytes", cached_tokens * kv_bytes_per_token()))
        remote_bytes = int(schedule.get("remote_context_bytes", max(0, total_context_tokens - cached_tokens) * kv_bytes_per_token()))
        context_domain_id = schedule.get("context_domain_id", ev.shared_prefix_id or ev.prompt_hash or ev.instance_id)
        self.stats["local_memory_bytes"] += local_bytes
        self.stats["remote_communication_bytes"] += remote_bytes
        last_dep = deps
        if local_bytes > 0:
            local_id = self._add(
                f"{ev.node_id}:local_cached_context",
                "memory",
                npu,
                seconds=local_bytes / 1e12,
                size_bytes=local_bytes,
                deps=last_dep,
                metadata={"event_id": ev.event_id, "policy": policy, "context_domain_id": context_domain_id, "mapping": "local cached context"},
            )
            last_dep = [local_id]
        if remote_bytes > 0:
            comm_id = self._add(
                f"{ev.node_id}:remote_context",
                "communication",
                npu,
                seconds=remote_bytes / 2e11,
                size_bytes=remote_bytes,
                deps=last_dep,
                metadata={"event_id": ev.event_id, "policy": policy, "context_domain_id": context_domain_id, "mapping": "remote context movement"},
            )
            mem_id = self._add(
                f"{ev.node_id}:remote_context_load",
                "memory",
                npu,
                seconds=remote_bytes / 1e12,
                size_bytes=remote_bytes,
                deps=[comm_id],
                metadata={"event_id": ev.event_id, "policy": policy, "context_domain_id": context_domain_id, "mapping": "remote context memory load"},
            )
            last_dep = [mem_id]
        prefill = self.lm.predict_prefill(max(0, ev.input_tokens - cached_tokens))
        decode = self.lm.predict_decode(ev.input_tokens, ev.output_tokens)
        prefill_id = self._add(
            f"{ev.node_id}:policy_prefill",
            "compute",
            npu,
            seconds=prefill,
            deps=last_dep,
            metadata={
                "event_id": ev.event_id,
                "policy": policy,
                "mapping": "LLM prefill after policy cache",
                "input_tokens": ev.input_tokens,
                "cached_tokens": cached_tokens,
            },
        )
        decode_id = self._add(
            f"{ev.node_id}:policy_decode",
            "compute",
            npu,
            seconds=decode,
            deps=[prefill_id],
            metadata={"event_id": ev.event_id, "policy": policy, "mapping": "LLM decode", "output_tokens": ev.output_tokens},
        )
        self._finish_event(ev, decode_id)

    def export_trace(self, trace: Trace, out_path: str | Path) -> dict[str, Any]:
        for ev in sorted(trace.events, key=lambda e: (e.branch_id, e.step_id, e.timestamp_start or 0.0, e.node_id)):
            if ev.node_type in {"llm", "tool", "verifier"}:
                self.add_event(ev)
        payload = {
            "format": "agentweaver_chakra_intermediate_json",
            "astra_export_format": "intermediate_json",
            "metadata": trace.metadata,
            "nodes": [n.to_dict() for n in self.nodes],
            "stats": self.stats,
        }
        path = Path(out_path)
        ensure_dir(path.parent)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    def export_policy_aware_trace(
        self,
        trace: Trace,
        out_path: str | Path,
        policy: str = "acd_nisp",
        schedule: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        schedule = schedule or infer_policy_schedule(trace, policy, self.npu_count)
        for ev in sorted(trace.events, key=lambda e: (e.branch_id, e.step_id, e.timestamp_start or 0.0, e.node_id)):
            if ev.node_type in {"llm", "tool", "verifier"}:
                self.add_event_policy_aware(ev, schedule.get(ev.event_id, {}), policy)
        payload = {
            "format": "agentweaver_chakra_intermediate_json",
            "astra_export_format": "intermediate_json",
            "policy_aware": True,
            "policy": policy,
            "metadata": trace.metadata,
            "nodes": [n.to_dict() for n in self.nodes],
            "stats": self.stats,
            "schedule": schedule,
        }
        path = Path(out_path)
        ensure_dir(path.parent)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload


def infer_policy_schedule(trace: Trace, policy: str = "acd_nisp", npu_count: int = 16) -> dict[str, dict[str, Any]]:
    schedule: dict[str, dict[str, Any]] = {}
    seen_domains: set[str] = set()
    for ev in sorted(trace.events, key=lambda e: (e.branch_id, e.step_id, e.timestamp_start or 0.0, e.node_id)):
        if ev.node_type != "llm":
            continue
        shared_tokens = sum(ref.length for ref in ev.context_segments if ref.segment_type in {"system", "tool_schema", "task", "repo", "history"})
        total_context_tokens = sum(ref.length for ref in ev.context_segments)
        domain = ev.shared_prefix_id or ev.prompt_hash or ev.instance_id
        region = _npu_for_event(ev, npu_count)
        if policy in {"naive_wafer", "raw"}:
            cached = 0
        elif policy in {"acd", "acd_nisp", "taps_c", "taps_unified_v5", "taps_unified_adaptive_v6"}:
            cached = shared_tokens if domain in seen_domains or policy in {"acd_nisp", "taps_c"} else max(0, shared_tokens // 2)
        else:
            cached = min(shared_tokens, int(ev.kv_cache_hit_tokens or 0))
        cached = min(cached, ev.input_tokens, total_context_tokens)
        schedule[ev.event_id] = {
            "cached_tokens": cached,
            "local_context_bytes": cached * kv_bytes_per_token(),
            "remote_context_bytes": max(0, total_context_tokens - cached) * kv_bytes_per_token(),
            "region_id": region,
            "context_domain_id": domain,
            "policy": policy,
        }
        seen_domains.add(domain)
    return schedule


def export_trace_to_chakra_json(
    trace_path: str | Path,
    out_path: str | Path,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    npu_count: int = 16,
) -> dict[str, Any]:
    trace = Trace.from_jsonl(trace_path)
    lm = LatencyModel.load(model_json)
    return ChakraExporter(lm, npu_count=npu_count).export_trace(trace, out_path)


def export_policy_aware_trace_to_chakra_json(
    trace_path: str | Path,
    out_path: str | Path,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    npu_count: int = 16,
    policy: str = "acd_nisp",
    schedule_json: str | Path | None = None,
) -> dict[str, Any]:
    trace = Trace.from_jsonl(trace_path)
    lm = LatencyModel.load(model_json)
    schedule = None
    if schedule_json and Path(schedule_json).exists():
        schedule = json.loads(Path(schedule_json).read_text(encoding="utf-8"))
    return ChakraExporter(lm, npu_count=npu_count).export_policy_aware_trace(trace, out_path, policy, schedule)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-json", default="data/profiles/h100_latency_model_pr2_v2.json")
    ap.add_argument("--npu-count", type=int, default=16)
    ap.add_argument("--policy-aware", action="store_true")
    ap.add_argument("--policy", default="acd_nisp")
    ap.add_argument("--schedule-json")
    args = ap.parse_args()
    if args.policy_aware:
        payload = export_policy_aware_trace_to_chakra_json(args.trace, args.out, args.model_json, args.npu_count, args.policy, args.schedule_json)
    else:
        payload = export_trace_to_chakra_json(args.trace, args.out, args.model_json, args.npu_count)
    print(json.dumps(payload["stats"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
