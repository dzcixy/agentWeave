from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agentweaver.analysis.context_segment_graph import kv_bytes_per_token
from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.simulator.metrics import percentile
from agentweaver.simulator.multisession_replay import ToolLatencyPredictor
from agentweaver.simulator.taps_domain_scheduler import context_domain_id, repo_domain_id
from agentweaver.tracing.trace_schema import Event, Trace, load_trace_dir
from agentweaver.utils.io import ensure_dir, write_csv


POLICIES = ["static_admission", "reactive_admission", "taps_admission"]


@dataclass
class AdmissionConfig:
    w_reuse: float = 2.0
    w_llm: float = 0.5
    w_stall: float = 1.0
    w_sram: float = 0.5
    w_tool_penalty: float = 0.5


@dataclass(order=True)
class SimEvent:
    time: float
    seq: int
    kind: str = field(compare=False)
    sid: str = field(compare=False)
    event: Event | None = field(default=None, compare=False)
    region_id: int | None = field(default=None, compare=False)


@dataclass
class SessionState:
    sid: str
    template_id: str
    arrival_time: float
    events: list[Event]
    domain: str
    repo: str
    next_index: int = 0
    done_time: float = 0.0
    previous_tool_latency: float | None = None
    predicted_tool_return_time: float = 0.0
    parked_cached_tokens: int = 0


def _events(trace: Trace) -> list[Event]:
    return sorted(
        [e for e in trace.events if e.branch_id != "root" and e.node_type in {"llm", "tool", "verifier"}],
        key=lambda e: (e.step_id, e.timestamp_start or e.timestamp_ready or 0.0, e.node_id),
    )


def _tool_latency(ev: Event | None) -> float:
    if ev is None:
        return 0.0
    return max(0.0, float(ev.tool_latency if ev.tool_latency is not None else ev.latency or 0.0))


class AdmissionReplay:
    def __init__(
        self,
        traces: list[Trace],
        total_sessions: int,
        active_session_limit: int,
        effective_regions: int,
        policy: str,
        latency_model: LatencyModel,
        config: AdmissionConfig | None = None,
        seed: int = 71,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy}")
        self.traces = traces
        self.total_sessions = total_sessions
        self.limit = max(1, active_session_limit)
        self.regions = max(1, effective_regions)
        self.policy = policy
        self.lm = latency_model
        self.config = config or AdmissionConfig()
        self.rng = random.Random(seed)
        self.bpt = kv_bytes_per_token()
        self.seq = 0
        self.eventq: list[SimEvent] = []
        self.ready: deque[tuple[str, Event]] = deque()
        self.ready_since: dict[tuple[str, str], float] = {}
        self.free_regions = list(range(self.regions))
        self.backlog: list[tuple[str, Trace, list[Event], str, str]] = []
        self.active: dict[str, SessionState] = {}
        self.done: list[SessionState] = []
        self.resident_domains: set[str] = set()
        self.predictor = ToolLatencyPredictor()
        self.predictor.train([e for tr in traces for e in tr.events])
        self.llm_busy = 0.0
        self.tool_time = 0.0
        self.ready_wait = 0.0
        self.ready_empty_time = 0.0
        self.last_event_time = 0.0
        self.blocked_time_area = 0.0
        self.admission_count = 0
        self.eviction_count = 0

    def _push(self, t: float, kind: str, sid: str, event: Event | None = None, region_id: int | None = None) -> None:
        self.seq += 1
        heapq.heappush(self.eventq, SimEvent(t, self.seq, kind, sid, event, region_id))

    def _prepare_backlog(self) -> None:
        for i in range(self.total_sessions):
            tr = self.traces[i % len(self.traces)]
            evs = _events(tr)
            if not evs:
                continue
            first = next((e for e in evs if e.node_type == "llm"), evs[0])
            self.backlog.append((f"s{i}", tr, evs, context_domain_id(first), repo_domain_id(first.instance_id)))

    def _next(self, st: SessionState) -> Event | None:
        if st.next_index >= len(st.events):
            return None
        ev = st.events[st.next_index]
        st.next_index += 1
        return ev

    def _peek(self, st: SessionState) -> Event | None:
        return st.events[st.next_index] if st.next_index < len(st.events) else None

    def _candidate_score(self, cand: tuple[str, Trace, list[Event], str, str]) -> float:
        _, _, evs, domain, repo = cand
        llm = [e for e in evs if e.node_type == "llm"]
        tools = [e for e in evs if e.node_type == "tool"]
        prompt_tokens = sum(e.input_tokens for e in llm)
        reuse = 1.0 if domain in self.resident_domains else 0.0
        expected_llm = min(1.0, prompt_tokens / 100000.0)
        predicted_tools = [self.predictor.predict(e) for e in tools[:3]]
        stall_overlap = min(1.0, sum(predicted_tools) / 30.0)
        tool_penalty = min(1.0, sum(predicted_tools) / max(1e-9, sum(predicted_tools) + len(llm)))
        sram_pressure = len(self.resident_domains) / 64.0
        return (
            self.config.w_reuse * reuse
            + self.config.w_llm * expected_llm
            + self.config.w_stall * stall_overlap
            - self.config.w_sram * sram_pressure
            - self.config.w_tool_penalty * tool_penalty
        )

    def _admit_one(self, now: float) -> bool:
        if not self.backlog or len(self.active) >= self.limit:
            return False
        if self.policy == "taps_admission":
            idx = max(range(len(self.backlog)), key=lambda i: self._candidate_score(self.backlog[i]))
        else:
            idx = 0
        sid, tr, evs, domain, repo = self.backlog.pop(idx)
        st = SessionState(sid, str(tr.metadata.get("source", "")), now, evs, domain, repo)
        self.active[sid] = st
        self.admission_count += 1
        self._push(now, "SESSION_ARRIVAL", sid)
        return True

    def _admit_initial(self) -> None:
        self._prepare_backlog()
        initial = min(self.limit, len(self.backlog))
        for _ in range(initial):
            self._admit_one(0.0)

    def _maybe_admit(self, now: float) -> None:
        if self.policy == "static_admission":
            if not self.active and self.backlog:
                for _ in range(min(self.limit, len(self.backlog))):
                    self._admit_one(now)
            return
        while self.backlog and len(self.active) < self.limit and len(self.ready) < self.regions:
            if not self._admit_one(now):
                break

    def _schedule(self, now: float) -> None:
        while self.ready and self.free_regions:
            sid, ev = self.ready.popleft()
            region = self.free_regions.pop(0)
            self.ready_wait += max(0.0, now - self.ready_since.pop((sid, ev.event_id), now))
            self._push(now, "LLM_START", sid, ev, region)

    def _enqueue_next(self, now: float, st: SessionState) -> None:
        nxt = self._next(st)
        if nxt is None:
            st.done_time = now
            self.done.append(st)
            self.active.pop(st.sid, None)
            self._maybe_admit(now)
        elif nxt.node_type == "llm":
            self.ready.append((st.sid, nxt))
            self.ready_since[(st.sid, nxt.event_id)] = now
            self._schedule(now)
        elif nxt.node_type == "tool":
            self._push(now, "TOOL_START", st.sid, nxt)
        elif nxt.node_type == "verifier":
            self._push(now + max(0.0, float(nxt.latency or 0.0)), "SESSION_DONE", st.sid, nxt)

    def _account_idle(self, now: float) -> None:
        dt = max(0.0, now - self.last_event_time)
        if not self.ready and self.free_regions:
            self.ready_empty_time += dt
        if self.active:
            blocked = sum(1 for st in self.active.values() if st.predicted_tool_return_time > self.last_event_time)
            self.blocked_time_area += blocked * dt
        self.last_event_time = now

    def run(self) -> dict[str, Any]:
        self._admit_initial()
        last = 0.0
        while self.eventq:
            item = heapq.heappop(self.eventq)
            now = item.time
            self._account_idle(now)
            last = max(last, now)
            st = self.active.get(item.sid)
            if item.kind == "SESSION_ARRIVAL" and st is not None:
                self._enqueue_next(now, st)
            elif item.kind == "LLM_START" and st is not None and item.event is not None:
                cached = st.parked_cached_tokens if self.policy != "static_admission" else 0
                st.parked_cached_tokens = 0
                dur = self.lm.predict_llm(item.event.input_tokens, item.event.output_tokens, min(cached, item.event.input_tokens))
                self.llm_busy += dur
                self._push(now + dur, "LLM_DONE", item.sid, item.event, item.region_id)
            elif item.kind == "LLM_DONE" and st is not None:
                if item.event is not None:
                    self.resident_domains.add(st.domain)
                if item.region_id is not None:
                    self.free_regions.append(item.region_id)
                    self.free_regions.sort()
                self._enqueue_next(now, st)
                self._maybe_admit(now)
                self._schedule(now)
            elif item.kind == "TOOL_START" and st is not None:
                lat = _tool_latency(item.event)
                self.tool_time += lat
                st.predicted_tool_return_time = now + (self.predictor.predict(item.event, st) if item.event else 0.0)
                nxt = self._peek(st)
                if nxt is not None and nxt.node_type == "llm" and self.policy != "static_admission":
                    st.parked_cached_tokens = sum(ref.length for ref in nxt.context_segments if ref.segment_type in {"system", "tool_schema", "task", "repo", "history"})
                self._maybe_admit(now)
                self._push(now + lat, "TOOL_DONE", item.sid, item.event)
            elif item.kind == "TOOL_DONE" and st is not None:
                st.predicted_tool_return_time = 0.0
                self._enqueue_next(now, st)
                self._schedule(now)
            elif item.kind == "SESSION_DONE" and st is not None:
                st.done_time = now
                self.done.append(st)
                self.active.pop(st.sid, None)
                self._maybe_admit(now)
        jcts = [st.done_time - st.arrival_time for st in self.done]
        makespan = max([st.done_time for st in self.done] or [last, 1e-9])
        total_region_time = max(1e-9, makespan * self.regions)
        return {
            "total_sessions": self.total_sessions,
            "active_session_limit": self.limit,
            "effective_regions": self.regions,
            "policy": self.policy,
            "throughput": len(self.done) / max(1e-9, makespan),
            "mean_jct": sum(jcts) / max(1, len(jcts)),
            "p95_jct": percentile(jcts, 95),
            "p99_jct": percentile(jcts, 99),
            "region_utilization": self.llm_busy / total_region_time,
            "ready_queue_empty_time": self.ready_empty_time,
            "blocked_session_fraction": self.blocked_time_area / max(1e-9, makespan * max(1, self.limit)),
            "completed_sessions": len(self.done),
            "admission_count": self.admission_count,
            "eviction_count": self.eviction_count,
        }


def _fill_gains(rows: list[dict[str, Any]]) -> None:
    by_key: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(int(row["total_sessions"]), int(row["active_session_limit"]), int(row["effective_regions"]))][row["policy"]] = row
    for group in by_key.values():
        base = group.get("static_admission")
        taps = group.get("taps_admission")
        if not base or not taps:
            continue
        for row in group.values():
            row["admission_throughput_gain"] = (float(taps["throughput"]) - float(base["throughput"])) / max(1e-9, float(base["throughput"]))
            row["admission_p95_jct_gain"] = (float(base["p95_jct"]) - float(taps["p95_jct"])) / max(1e-9, float(base["p95_jct"]))
            row["admission_region_util_gain"] = float(taps["region_utilization"]) - float(base["region_utilization"])


def run_admission(
    trace_dirs: list[str | Path] | None = None,
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
    out_csv: str | Path = "data/results/taps_admission_pr4_v4.csv",
    total_sessions_list: list[int] | None = None,
    active_limits: list[int] | None = None,
    effective_regions_list: list[int] | None = None,
    config: AdmissionConfig | None = None,
) -> list[dict[str, Any]]:
    trace_dirs = trace_dirs or ["data/traces/mini_swe_lite10_r4_timed", "data/traces/mini_swe_lite5_patchcap_verified"]
    traces: list[Trace] = []
    for d in trace_dirs:
        if Path(d).exists():
            traces.extend(load_trace_dir(d))
    lm = LatencyModel.load(model_json)
    total_sessions_list = total_sessions_list or [16, 32, 64, 128]
    active_limits = active_limits or [4, 8, 16, 32]
    effective_regions_list = effective_regions_list or [1, 2, 4, 8, 16]
    rows: list[dict[str, Any]] = []
    for total, limit, regions in itertools.product(total_sessions_list, active_limits, effective_regions_list):
        if limit > total:
            continue
        for policy in POLICIES:
            rows.append(AdmissionReplay(traces, total, limit, regions, policy, lm, config, seed=83 + total + regions).run())
    _fill_gains(rows)
    write_csv(out_csv, rows)
    plot_admission(rows)
    return rows


def plot_admission(rows: list[dict[str, Any]], out: str | Path = "data/plots/taps_admission_pr4_v4.pdf") -> None:
    ensure_dir(Path(out).parent)
    sub = [r for r in rows if int(r.get("active_session_limit", 0) or 0) == 16 and int(r.get("effective_regions", 0) or 0) == 4]
    totals = sorted({int(r["total_sessions"]) for r in sub})
    plt.figure(figsize=(6.4, 3.8))
    for policy in POLICIES:
        vals = [next((float(r["p95_jct"]) for r in sub if r["policy"] == policy and int(r["total_sessions"]) == t), 0.0) for t in totals]
        plt.plot(totals, vals, marker="o", label=policy)
    plt.xlabel("total sessions")
    plt.ylabel("p95 JCT (s)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/taps_admission_pr4_v4.csv")
    args = ap.parse_args()
    print(json.dumps({"rows": len(run_admission(out_csv=args.out)), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
