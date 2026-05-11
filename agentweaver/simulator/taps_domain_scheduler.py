from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.analysis.context_segment_graph import kv_bytes_per_token
from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.metrics import percentile
from agentweaver.simulator.multisession_replay import ToolLatencyPredictor
from agentweaver.tracing.trace_schema import ContextSegmentRef, Event, Trace, load_trace_dir
from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.io import ensure_dir, write_csv


POLICIES = ["naive_wafer", "acd_nisp", "taps_domain"]


@dataclass
class DomainConfig:
    w_domain: float = 2.0
    w_batch: float = 0.5
    w_hot: float = 0.3
    w_resume: float = 0.5
    w_age: float = 0.05
    w_short: float = 0.3
    w_remote: float = 0.5
    w_evict: float = 0.0
    tau: float = 10.0


@dataclass(order=True)
class SimEvent:
    time: float
    seq: int
    kind: str = field(compare=False)
    session_id: str = field(compare=False)
    event: Event | None = field(default=None, compare=False)
    region_id: int | None = field(default=None, compare=False)


@dataclass
class SessionState:
    session_id: str
    trace: Trace
    arrival_time: float
    events: list[Event]
    next_index: int = 0
    done_time: float = 0.0
    predicted_tool_return_time: float = 0.0
    previous_tool_latency: float | None = None
    context_domain_id: str = ""
    repo_domain_id: str = ""
    current_region: int | None = None
    parked_cached_tokens: int = 0


def repo_domain_id(instance_id: str) -> str:
    if "__" in instance_id:
        return instance_id.split("__", 1)[0]
    return instance_id.split("-", 1)[0] if "-" in instance_id else instance_id[:16]


def context_domain_id(ev: Event) -> str:
    pieces = [ev.instance_id]
    for ref in sorted(ev.context_segments, key=lambda r: (r.start_pos, r.segment_id)):
        if ref.segment_type in {"system", "tool_schema", "task", "repo"}:
            pieces.append(f"{ref.segment_type}:{ref.segment_id}:{ref.length}")
    return stable_hash("|".join(pieces))


def _session_events(trace: Trace) -> list[Event]:
    return sorted(
        [e for e in trace.events if e.branch_id != "root" and e.node_type in {"llm", "tool", "verifier"}],
        key=lambda e: (e.step_id, e.timestamp_start or e.timestamp_ready or 0.0, e.node_id),
    )


def _tool_latency(ev: Event | None) -> float:
    if ev is None:
        return 0.0
    return max(0.0, float(ev.tool_latency if ev.tool_latency is not None else ev.latency or 0.0))


def _arrival(i: int, pattern: str, regions: int, rng: random.Random) -> float:
    if pattern == "closed_loop":
        return 0.0
    if pattern == "bursty":
        return (i // max(1, regions)) * 0.6 + (i % max(1, regions)) * 0.015
    rate = max(0.25, min(6.0, regions / 1.5))
    return sum(rng.expovariate(rate) for _ in range(i + 1))


class DomainReplay:
    def __init__(
        self,
        traces: list[Trace],
        sessions: int,
        policy: str,
        latency_model: LatencyModel,
        effective_regions: int,
        arrival_pattern: str,
        config: DomainConfig | None = None,
        seed: int = 31,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy}")
        self.traces = traces
        self.sessions_n = sessions
        self.policy = policy
        self.lm = latency_model
        self.regions = max(1, effective_regions)
        self.arrival_pattern = arrival_pattern
        self.config = config or DomainConfig()
        self.rng = random.Random(seed)
        self.bpt = kv_bytes_per_token()
        self.seq = 0
        self.eventq: list[SimEvent] = []
        self.ready: deque[tuple[str, Event]] = deque()
        self.ready_since: dict[tuple[str, str], float] = {}
        self.free_regions = list(range(self.regions))
        self.sessions: dict[str, SessionState] = {}
        self.resident_segments: dict[str, set[str]] = defaultdict(set)
        self.domain_home_region: dict[str, int] = {}
        self.domain_hotness: Counter[str] = Counter()
        self.last_domain: str | None = None
        self.predictor = ToolLatencyPredictor()
        self.predictor.train([ev for tr in traces for ev in tr.events])
        self.llm_busy = 0.0
        self.tool_time = 0.0
        self.ready_wait = 0.0
        self.remote_kv_bytes = 0.0
        self.hop_weighted_bytes = 0.0
        self.domain_cache_hit_tokens = 0
        self.domain_cache_query_tokens = 0
        self.domain_batch_sizes: list[int] = []
        self.domain_switches = 0

    def _push(self, time: float, kind: str, session_id: str, event: Event | None = None, region_id: int | None = None) -> None:
        self.seq += 1
        heapq.heappush(self.eventq, SimEvent(time, self.seq, kind, session_id, event, region_id))

    def _init_sessions(self) -> None:
        for i in range(self.sessions_n):
            trace = self.traces[i % len(self.traces)]
            events = _session_events(trace)
            if not events:
                continue
            first = next((e for e in events if e.node_type == "llm"), events[0])
            domain = context_domain_id(first)
            repo = repo_domain_id(first.instance_id)
            self.domain_home_region.setdefault(domain, int(stable_hash(domain), 16) % self.regions)
            sid = f"s{i}"
            arrival = _arrival(i, self.arrival_pattern, self.regions, self.rng)
            self.sessions[sid] = SessionState(sid, trace, arrival, events, context_domain_id=domain, repo_domain_id=repo)
            self._push(arrival, "SESSION_ARRIVAL", sid)

    def _next(self, st: SessionState) -> Event | None:
        if st.next_index >= len(st.events):
            return None
        ev = st.events[st.next_index]
        st.next_index += 1
        return ev

    def _peek(self, st: SessionState) -> Event | None:
        return st.events[st.next_index] if st.next_index < len(st.events) else None

    def _domain_ready_count(self, domain: str, repo: str) -> int:
        count = 0
        for sid, _ in self.ready:
            st = self.sessions[sid]
            if st.context_domain_id == domain or st.repo_domain_id == repo:
                count += 1
        return count

    def _resident_tokens(self, domain: str, refs: list[ContextSegmentRef]) -> int:
        resident = self.resident_segments.get(domain, set())
        return sum(ref.length for ref in refs if ref.segment_id in resident)

    def _remote_cost(self, domain: str, region: int, cached_tokens: int) -> tuple[float, float]:
        home = self.domain_home_region.setdefault(domain, int(stable_hash(domain), 16) % self.regions)
        hops = abs(region - home)
        bytes_ = cached_tokens * self.bpt
        return float(hops), bytes_

    def _score(self, now: float, sid: str, ev: Event) -> float:
        st = self.sessions[sid]
        domain = st.context_domain_id
        repo = st.repo_domain_id
        cached = min(ev.input_tokens, self._resident_tokens(domain, ev.context_segments) + st.parked_cached_tokens)
        domain_locality = cached / max(1, ev.input_tokens)
        batch_bonus = self._domain_ready_count(domain, repo)
        hotness = math.log1p(self.domain_hotness[domain])
        resume = math.exp(-max(0.0, st.predicted_tool_return_time - now) / max(1e-9, self.config.tau))
        age = now - self.ready_since.get((sid, ev.event_id), now)
        service = self.lm.predict_llm(ev.input_tokens, ev.output_tokens, cached)
        short = 1.0 / max(1e-9, service)
        home = self.domain_home_region.setdefault(domain, int(stable_hash(domain), 16) % self.regions)
        nearest = min(self.free_regions or [home], key=lambda r: abs(r - home))
        remote_penalty = abs(nearest - home) * cached * self.bpt / 1e9
        return (
            self.config.w_domain * domain_locality
            + self.config.w_batch * batch_bonus
            + self.config.w_hot * hotness
            + self.config.w_resume * resume
            + self.config.w_age * age
            + self.config.w_short * short
            - self.config.w_remote * remote_penalty
        )

    def _choose_region(self, st: SessionState) -> int:
        if self.policy == "taps_domain":
            home = self.domain_home_region.setdefault(st.context_domain_id, int(stable_hash(st.context_domain_id), 16) % self.regions)
            region = min(self.free_regions, key=lambda r: abs(r - home))
        else:
            region = self.free_regions[0]
        self.free_regions.remove(region)
        return region

    def _try_schedule(self, now: float) -> None:
        while self.ready and self.free_regions:
            if self.policy == "taps_domain":
                selected = max(self.ready, key=lambda pair: self._score(now, pair[0], pair[1]))
            else:
                selected = self.ready[0]
            self.ready.remove(selected)
            sid, ev = selected
            st = self.sessions[sid]
            region = self._choose_region(st)
            self.ready_wait += max(0.0, now - self.ready_since.pop((sid, ev.event_id), now))
            self._push(now, "LLM_START", sid, ev, region)

    def _enqueue_next(self, now: float, st: SessionState) -> None:
        nxt = self._next(st)
        if nxt is None:
            st.done_time = now
        elif nxt.node_type == "llm":
            self.ready.append((st.session_id, nxt))
            self.ready_since[(st.session_id, nxt.event_id)] = now
            self._try_schedule(now)
        elif nxt.node_type == "tool":
            self._push(now, "TOOL_START", st.session_id, nxt)
        elif nxt.node_type == "verifier":
            self._push(now + max(0.0, float(nxt.latency or 0.0)), "SESSION_DONE", st.session_id, nxt)

    def _llm_duration(self, st: SessionState, ev: Event, region: int) -> float:
        if self.policy == "naive_wafer":
            cached = 0
        else:
            cached = min(ev.input_tokens, self._resident_tokens(st.context_domain_id, ev.context_segments) + st.parked_cached_tokens)
        st.parked_cached_tokens = 0
        self.domain_cache_hit_tokens += cached
        self.domain_cache_query_tokens += ev.input_tokens
        hops, bytes_ = self._remote_cost(st.context_domain_id, region, cached)
        self.remote_kv_bytes += bytes_ * hops
        self.hop_weighted_bytes += bytes_ * hops
        remote_penalty = (bytes_ * hops) / 2e11
        return self.lm.predict_llm(ev.input_tokens, ev.output_tokens, cached) + remote_penalty

    def run(self) -> dict[str, Any]:
        self._init_sessions()
        last_time = 0.0
        while self.eventq:
            item = heapq.heappop(self.eventq)
            now = item.time
            last_time = max(last_time, now)
            st = self.sessions.get(item.session_id)
            if st is None:
                continue
            if item.kind == "SESSION_ARRIVAL":
                self._enqueue_next(now, st)
            elif item.kind == "LLM_START":
                if item.event is None or item.region_id is None:
                    continue
                if self.last_domain is not None and self.last_domain != st.context_domain_id:
                    self.domain_switches += 1
                self.last_domain = st.context_domain_id
                self.domain_batch_sizes.append(self._domain_ready_count(st.context_domain_id, st.repo_domain_id) + 1)
                dur = self._llm_duration(st, item.event, item.region_id)
                self.llm_busy += dur
                st.current_region = item.region_id
                self._push(now + dur, "LLM_DONE", item.session_id, item.event, item.region_id)
            elif item.kind == "LLM_DONE":
                if item.event is not None:
                    self.domain_hotness[st.context_domain_id] += 1
                    for ref in item.event.context_segments:
                        if self.policy != "naive_wafer":
                            self.resident_segments[st.context_domain_id].add(ref.segment_id)
                if item.region_id is not None:
                    self.free_regions.append(item.region_id)
                    self.free_regions.sort()
                self._enqueue_next(now, st)
                self._try_schedule(now)
            elif item.kind == "TOOL_START":
                lat = _tool_latency(item.event)
                self.tool_time += lat
                if item.event is not None:
                    st.previous_tool_latency = lat
                    st.predicted_tool_return_time = now + self.predictor.predict(item.event, st)
                nxt = self._peek(st)
                if nxt is not None and nxt.node_type == "llm" and self.policy != "naive_wafer":
                    st.parked_cached_tokens = sum(ref.length for ref in nxt.context_segments if ref.segment_type in {"system", "tool_schema", "task", "repo", "history"})
                self._push(now + lat, "TOOL_DONE", item.session_id, item.event)
                self._try_schedule(now)
            elif item.kind == "TOOL_DONE":
                self._enqueue_next(now, st)
                self._try_schedule(now)
            elif item.kind == "SESSION_DONE":
                st.done_time = now
        completed = [s for s in self.sessions.values() if s.done_time > 0]
        jcts = [s.done_time - s.arrival_time for s in completed]
        makespan = max([s.done_time for s in completed] or [last_time, 1e-9])
        total_region_time = max(1e-9, makespan * self.regions)
        remote_bytes = self.remote_kv_bytes
        return {
            "policy": self.policy,
            "sessions": self.sessions_n,
            "arrival_pattern": self.arrival_pattern,
            "effective_regions": self.regions,
            "throughput_sessions_per_sec": len(completed) / max(1e-9, makespan),
            "mean_jct": sum(jcts) / max(1, len(jcts)),
            "p50_jct": percentile(jcts, 50),
            "p95_jct": percentile(jcts, 95),
            "p99_jct": percentile(jcts, 99),
            "ready_queue_wait": self.ready_wait,
            "region_utilization": self.llm_busy / total_region_time,
            "domain_cache_hit_rate": self.domain_cache_hit_tokens / max(1, self.domain_cache_query_tokens),
            "remote_kv_bytes": remote_bytes,
            "avg_context_hops": self.hop_weighted_bytes / max(1.0, remote_bytes),
            "domain_batch_size": sum(self.domain_batch_sizes) / max(1, len(self.domain_batch_sizes)),
            "domain_switches": self.domain_switches,
            "completed_sessions": len(completed),
        }


def _fill_gains(rows: list[dict[str, Any]]) -> None:
    by_key: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(row["arrival_pattern"], int(row["effective_regions"]), int(row["sessions"]))][row["policy"]] = row
    for group in by_key.values():
        base = group.get("acd_nisp")
        dom = group.get("taps_domain")
        if not base or not dom:
            continue
        for row in group.values():
            row["domain_remote_kv_reduction"] = (float(base["remote_kv_bytes"]) - float(dom["remote_kv_bytes"])) / max(1.0, float(base["remote_kv_bytes"]))
            row["domain_p95_jct_gain"] = (float(base["p95_jct"]) - float(dom["p95_jct"])) / max(1e-9, float(base["p95_jct"]))
            row["domain_mean_jct_gain"] = (float(base["mean_jct"]) - float(dom["mean_jct"])) / max(1e-9, float(base["mean_jct"]))
            row["domain_ready_wait_gain"] = (float(base["ready_queue_wait"]) - float(dom["ready_queue_wait"])) / max(1e-9, float(base["ready_queue_wait"]))


def run_domain_scheduler(
    trace_dir: str | Path = "data/traces/mini_swe_lite10_r4_timed",
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    out_csv: str | Path = "data/results/taps_domain_scheduler_pr4_v4.csv",
    sessions_list: list[int] | None = None,
    effective_regions_list: list[int] | None = None,
    arrival_patterns: list[str] | None = None,
    config: DomainConfig | None = None,
) -> list[dict[str, Any]]:
    traces = load_trace_dir(trace_dir)
    lm = LatencyModel.load(model_json)
    sessions_list = sessions_list or [8, 16, 32, 64]
    effective_regions_list = effective_regions_list or [1, 2, 4, 8, 16]
    arrival_patterns = arrival_patterns or ["closed_loop", "poisson", "bursty"]
    rows: list[dict[str, Any]] = []
    for arrival, regions, sessions in itertools.product(arrival_patterns, effective_regions_list, sessions_list):
        for policy in POLICIES:
            rows.append(DomainReplay(traces, sessions, policy, lm, regions, arrival, config, seed=41 + sessions + regions).run())
    _fill_gains(rows)
    write_csv(out_csv, rows)
    plot_domain(rows)
    return rows


def plot_domain(rows: list[dict[str, Any]], out: str | Path = "data/plots/taps_domain_scheduler_pr4_v4.pdf") -> None:
    ensure_dir(Path(out).parent)
    sub = [r for r in rows if r.get("arrival_pattern") == "bursty" and int(r.get("effective_regions", 0) or 0) == 4]
    sessions = sorted({int(r["sessions"]) for r in sub})
    plt.figure(figsize=(6.4, 3.8))
    for policy in POLICIES:
        vals = [next((float(r["p95_jct"]) for r in sub if r["policy"] == policy and int(r["sessions"]) == s), 0.0) for s in sessions]
        plt.plot(sessions, vals, marker="o", label=policy)
    plt.xlabel("sessions")
    plt.ylabel("p95 JCT (s)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default="data/traces/mini_swe_lite10_r4_timed")
    ap.add_argument("--model-json", default="data/profiles/h100_latency_model_pr2_v2.json")
    ap.add_argument("--out", default="data/results/taps_domain_scheduler_pr4_v4.csv")
    args = ap.parse_args()
    print(json.dumps({"rows": len(run_domain_scheduler(args.trace_dir, args.model_json, args.out)), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
