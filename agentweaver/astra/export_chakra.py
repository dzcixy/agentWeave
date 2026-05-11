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
            "schedule_llm_events": 0,
            "schedule_cached_tokens": 0,
            "schedule_local_context_bytes": 0,
            "schedule_remote_context_bytes": 0,
            "schedule_match_error": 0.0,
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
        total_context_bytes = int(total_context_tokens * kv_bytes_per_token())
        cached_context_bytes = int(cached_tokens * kv_bytes_per_token())
        local_bytes = min(int(schedule.get("local_context_bytes", cached_context_bytes)), cached_context_bytes, total_context_bytes)
        remote_bytes = min(
            int(schedule.get("remote_context_bytes", 0)),
            max(0, cached_context_bytes - local_bytes),
            max(0, total_context_bytes - local_bytes),
        )
        context_domain_id = schedule.get("context_domain_id", ev.shared_prefix_id or ev.prompt_hash or ev.instance_id)
        self.stats["local_memory_bytes"] += local_bytes
        self.stats["remote_communication_bytes"] += remote_bytes
        if schedule:
            self.stats["schedule_llm_events"] += 1
            self.stats["schedule_cached_tokens"] += cached_tokens
            self.stats["schedule_local_context_bytes"] += local_bytes
            self.stats["schedule_remote_context_bytes"] += remote_bytes
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
        allow_inferred_schedule: bool = False,
    ) -> dict[str, Any]:
        schedule_source = "provided_schedule"
        warning = ""
        if schedule is None:
            if allow_inferred_schedule:
                schedule = infer_policy_schedule(trace, policy, self.npu_count)
                schedule_source = "inferred_schedule"
            else:
                schedule = {}
                schedule_source = "missing_schedule"
                warning = "WARNING_NO_SCHEDULE"
        for ev in sorted(trace.events, key=lambda e: (e.branch_id, e.step_id, e.timestamp_start or 0.0, e.node_id)):
            if ev.node_type in {"llm", "tool", "verifier"}:
                self.add_event_policy_aware(ev, schedule.get(ev.event_id, {}), policy)
        scheduled_remote = sum(int(row.get("remote_context_bytes", 0)) for row in schedule.values()) if schedule else 0
        self.stats["schedule_match_error"] = abs(float(self.stats["remote_communication_bytes"]) - float(scheduled_remote)) / max(1.0, float(scheduled_remote))
        payload = {
            "format": "agentweaver_chakra_intermediate_json",
            "astra_export_format": "intermediate_json",
            "policy_aware": True,
            "policy": policy,
            "schedule_source": schedule_source,
            "policy_aware_warning": warning,
            "metadata": trace.metadata,
            "nodes": [n.to_dict() for n in self.nodes],
            "stats": self.stats,
            "schedule": schedule,
        }
        path = Path(out_path)
        ensure_dir(path.parent)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    def export_schedule_jsonl(
        self,
        schedule_rows: list[dict[str, Any]],
        out_path: str | Path,
        *,
        policy: str,
        raw: bool = False,
    ) -> dict[str, Any]:
        schedule_remote = 0.0
        for row in sorted(schedule_rows, key=lambda r: (str(r.get("session_id", "")), float(r.get("timestamp_start", 0.0)), str(r.get("event_id", "")))):
            branch_key = f"{row.get('session_id', '')}:{row.get('branch_id', '')}"
            npu = int(float(row.get("region_id", 0))) % self.npu_count
            deps = [self.last_by_branch[branch_key]] if branch_key in self.last_by_branch else []
            input_tokens = int(float(row.get("input_tokens", 0) or 0))
            output_tokens = int(float(row.get("output_tokens", 0) or 0))
            cached_tokens = 0 if raw else min(input_tokens, int(float(row.get("cached_tokens", 0) or 0)))
            if raw:
                local_bytes = 0
                remote_bytes = int(input_tokens * kv_bytes_per_token())
            else:
                local_bytes = int(float(row.get("local_context_bytes", 0) or 0))
                remote_bytes = int(float(row.get("remote_kv_bytes", row.get("remote_context_bytes", 0)) or 0))
                schedule_remote += remote_bytes
            metadata = {
                "event_id": row.get("event_id", ""),
                "run_id": row.get("run_id", ""),
                "config_id": row.get("config_id", ""),
                "policy": "raw" if raw else policy,
                "context_domain_id": row.get("context_domain_id", ""),
                "schedule_source": "provided_schedule",
            }
            last_dep = deps
            if local_bytes > 0:
                local_id = self._add(
                    f"{row.get('node_id', row.get('event_id', 'llm'))}:local_context",
                    "memory",
                    npu,
                    seconds=local_bytes / 1e12,
                    size_bytes=local_bytes,
                    deps=last_dep,
                    metadata={**metadata, "mapping": "schedule local cached context"},
                )
                last_dep = [local_id]
                self.stats["local_memory_bytes"] += local_bytes
            if remote_bytes > 0:
                comm_id = self._add(
                    f"{row.get('node_id', row.get('event_id', 'llm'))}:remote_context",
                    "communication",
                    npu,
                    seconds=remote_bytes / 2e11,
                    size_bytes=remote_bytes,
                    deps=last_dep,
                    metadata={**metadata, "mapping": "schedule remote context"},
                )
                last_dep = [comm_id]
                self.stats["remote_communication_bytes"] += remote_bytes
            prefill = self.lm.predict_prefill(max(0, input_tokens - cached_tokens))
            decode = self.lm.predict_decode(input_tokens, output_tokens)
            prefill_id = self._add(
                f"{row.get('node_id', row.get('event_id', 'llm'))}:prefill",
                "compute",
                npu,
                seconds=prefill,
                deps=last_dep,
                metadata={**metadata, "mapping": "schedule LLM prefill", "input_tokens": input_tokens, "cached_tokens": cached_tokens},
            )
            decode_id = self._add(
                f"{row.get('node_id', row.get('event_id', 'llm'))}:decode",
                "compute",
                npu,
                seconds=decode,
                deps=[prefill_id],
                metadata={**metadata, "mapping": "schedule LLM decode", "output_tokens": output_tokens},
            )
            self.last_by_branch[branch_key] = decode_id
            self.stats["schedule_llm_events"] += 1
            self.stats["schedule_cached_tokens"] += cached_tokens
            self.stats["schedule_local_context_bytes"] += local_bytes
            self.stats["schedule_remote_context_bytes"] += remote_bytes
        if not raw:
            self.stats["schedule_match_error"] = abs(float(self.stats["remote_communication_bytes"]) - schedule_remote) / max(1.0, schedule_remote)
        payload = {
            "format": "agentweaver_chakra_intermediate_json",
            "astra_export_format": "intermediate_json",
            "policy_aware": not raw,
            "policy": "raw" if raw else policy,
            "schedule_source": "provided_schedule",
            "metadata": {
                "schedule_rows": len(schedule_rows),
                "config_id": schedule_rows[0].get("config_id", "") if schedule_rows else "",
                "run_id": schedule_rows[0].get("run_id", "") if schedule_rows else "",
            },
            "nodes": [n.to_dict() for n in self.nodes],
            "stats": self.stats,
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


def load_schedule_jsonl(path: str | Path) -> dict[str, dict[str, Any]]:
    schedule: dict[str, dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return schedule
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            event_id = row.get("event_id")
            if event_id:
                schedule[str(event_id)] = row
    return schedule


def load_schedule_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_per_npu_traces(payload: dict[str, Any], out_dir: str | Path, prefix: str = "agentweaver") -> dict[str, Any]:
    nodes = payload.get("nodes", [])
    by_npu: dict[int, list[dict[str, Any]]] = {}
    for node in nodes:
        npu = int(node.get("npu_id", 0))
        by_npu.setdefault(npu, []).append(node)
    out = Path(out_dir)
    ensure_dir(out)
    files: list[str] = []
    for npu, npu_nodes in sorted(by_npu.items()):
        p = out / f"{prefix}.{npu}.et.json"
        shard = {
            "format": payload.get("format", "agentweaver_chakra_intermediate_json"),
            "astra_export_format": payload.get("astra_export_format", "intermediate_json"),
            "npu_id": npu,
            "nodes": npu_nodes,
            "stats": {
                "node_count": len(npu_nodes),
                "communication_nodes": sum(1 for n in npu_nodes if n.get("type") == "communication"),
                "memory_nodes": sum(1 for n in npu_nodes if n.get("type") == "memory"),
                "compute_nodes": sum(1 for n in npu_nodes if n.get("type") == "compute"),
                "delay_nodes": sum(1 for n in npu_nodes if n.get("type") == "delay"),
            },
        }
        p.write_text(json.dumps(shard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files.append(str(p))
    return {
        "npu_files": files,
        "npu_file_count": len(files),
        "global_node_count": len(nodes),
        "per_npu_node_count_sum": sum(len(v) for v in by_npu.values()),
        "cross_npu_communication_nodes": sum(1 for n in nodes if n.get("type") == "communication"),
    }


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
    schedule_jsonl: str | Path | None = None,
    allow_inferred_schedule: bool = False,
) -> dict[str, Any]:
    trace = Trace.from_jsonl(trace_path)
    lm = LatencyModel.load(model_json)
    schedule = None
    if schedule_jsonl and Path(schedule_jsonl).exists():
        schedule = load_schedule_jsonl(schedule_jsonl)
    elif schedule_json and Path(schedule_json).exists():
        schedule = json.loads(Path(schedule_json).read_text(encoding="utf-8"))
    return ChakraExporter(lm, npu_count=npu_count).export_policy_aware_trace(trace, out_path, policy, schedule, allow_inferred_schedule)


def export_schedule_jsonl_to_chakra_json(
    schedule_jsonl: str | Path,
    out_path: str | Path,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    npu_count: int = 16,
    policy: str = "schedule_policy",
    raw: bool = False,
) -> dict[str, Any]:
    rows = load_schedule_jsonl_rows(schedule_jsonl)
    lm = LatencyModel.load(model_json)
    return ChakraExporter(lm, npu_count=npu_count).export_schedule_jsonl(rows, out_path, policy=policy, raw=raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-json", default="data/profiles/h100_latency_model_pr2_v2.json")
    ap.add_argument("--npu-count", type=int, default=16)
    ap.add_argument("--policy-aware", action="store_true")
    ap.add_argument("--policy", default="acd_nisp")
    ap.add_argument("--schedule-json")
    ap.add_argument("--schedule-jsonl")
    ap.add_argument("--allow-inferred-schedule", action="store_true")
    ap.add_argument("--schedule-only", action="store_true")
    ap.add_argument("--raw-schedule", action="store_true")
    ap.add_argument("--per-npu-dir")
    ap.add_argument("--per-npu-prefix", default="agentweaver")
    args = ap.parse_args()
    if args.schedule_only:
        payload = export_schedule_jsonl_to_chakra_json(
            args.schedule_jsonl or args.schedule_json or "",
            args.out,
            args.model_json,
            args.npu_count,
            args.policy,
            raw=args.raw_schedule,
        )
    elif args.policy_aware:
        payload = export_policy_aware_trace_to_chakra_json(
            args.trace,
            args.out,
            args.model_json,
            args.npu_count,
            args.policy,
            args.schedule_json,
            args.schedule_jsonl,
            args.allow_inferred_schedule,
        )
    else:
        payload = export_trace_to_chakra_json(args.trace, args.out, args.model_json, args.npu_count)
    if args.per_npu_dir:
        payload["per_npu"] = write_per_npu_traces(payload, args.per_npu_dir, args.per_npu_prefix)
    print(json.dumps(payload["stats"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
