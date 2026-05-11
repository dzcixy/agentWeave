from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentweaver.astra.export_chakra import ChakraExporter, export_trace_to_chakra_json
from agentweaver.astra.export_configs import write_smoke_configs
from agentweaver.profiling.latency_model import LatencyModel
from agentweaver.tracing.trace_schema import ContextSegmentRef, Event, Trace
from agentweaver.utils.io import ensure_dir


def _smoke_trace() -> Trace:
    segs = [
        ContextSegmentRef("system_smoke", "system", 0, 128),
        ContextSegmentRef("repo_smoke", "repo", 128, 512),
    ]
    return Trace(
        metadata={"source": "agentweaver_astra_smoke_fixture", "sample_fixture": True, "experimental_data": False},
        events=[
            Event(
                event_id="smoke:llm0",
                session_id="smoke",
                instance_id="smoke",
                branch_id="b0",
                parent_branch_id="root",
                step_id=0,
                node_id="smoke:llm0",
                node_type="llm",
                input_tokens=640,
                output_tokens=64,
                context_segments=segs,
            ),
            Event(
                event_id="smoke:tool0",
                session_id="smoke",
                instance_id="smoke",
                branch_id="b0",
                parent_branch_id="root",
                step_id=1,
                node_id="smoke:tool0",
                node_type="tool",
                tool_type="shell_other",
                command="pytest -q",
                tool_latency=1.25,
                latency=1.25,
            ),
            Event(
                event_id="smoke:llm1",
                session_id="smoke",
                instance_id="smoke",
                branch_id="b0",
                parent_branch_id="root",
                step_id=2,
                node_id="smoke:llm1",
                node_type="llm",
                input_tokens=960,
                output_tokens=80,
                context_segments=segs + [ContextSegmentRef("obs_smoke", "observation", 640, 320)],
            ),
        ],
    )


def _write_report(path: str | Path, title: str, payload: dict, extra: dict | None = None) -> None:
    stats = payload.get("stats", {})
    lines = [
        f"# {title}",
        "",
        "ASTRA_EXPORT_FORMAT = intermediate_json",
        "ASTRA_SIM_RUN_COMPLETED = false",
        f"COMPUTE_NODES = {stats.get('compute_nodes', 0)}",
        f"COMMUNICATION_NODES = {stats.get('communication_nodes', 0)}",
        f"MEMORY_NODES = {stats.get('memory_nodes', 0)}",
        f"DELAY_NODES = {stats.get('delay_nodes', 0)}",
        f"DEPENDENCY_COUNT = {stats.get('dependency_count', 0)}",
        f"ESTIMATED_COMMUNICATION_BYTES = {stats.get('estimated_communication_bytes', 0)}",
        f"ESTIMATED_COMPUTE_TIME = {stats.get('estimated_compute_time', 0.0)}",
    ]
    for k, v in (extra or {}).items():
        lines.append(f"{k} = {v}")
    lines.extend(["", "This is an AgentWeaver intermediate export, not an ASTRA-sim result."])
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_smoke(model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json") -> dict:
    lm = LatencyModel.load(model_json)
    write_smoke_configs("data/astra_configs/smoke")
    payload = ChakraExporter(lm, npu_count=16).export_trace(_smoke_trace(), "data/astra_traces/smoke/agentweaver_smoke.0.et.json")
    _write_report("data/results/astra_smoke_export_report.md", "ASTRA Smoke Export Report", payload)
    return payload


def export_one_real(
    trace_dir: str | Path = "data/traces/mini_swe_lite10_r4_timed",
    model_json: str | Path = "data/profiles/h100_latency_model_pr2_v2.json",
) -> dict | None:
    traces = sorted(Path(trace_dir).glob("*.jsonl"))
    if not traces:
        _write_report(
            "data/results/astra_export_mini_swe_report.md",
            "ASTRA mini-SWE Export Report",
            {"stats": {}},
            {"TRACE_FOUND": "false"},
        )
        return None
    write_smoke_configs("data/astra_configs/mini_swe_taps")
    out = Path("data/astra_traces/mini_swe_taps") / f"{traces[0].stem}.0.et.json"
    payload = export_trace_to_chakra_json(traces[0], out, model_json, 16)
    _write_report(
        "data/results/astra_export_mini_swe_report.md",
        "ASTRA mini-SWE Export Report",
        payload,
        {"TRACE_FOUND": "true", "TRACE_SOURCE": str(traces[0]), "TRACE_OUT": str(out)},
    )
    return payload


def run_all() -> dict:
    smoke = export_smoke()
    real = export_one_real()
    return {"smoke_nodes": len(smoke.get("nodes", [])), "real_nodes": len(real.get("nodes", [])) if real else 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default="data/traces/mini_swe_lite10_r4_timed")
    ap.add_argument("--model-json", default="data/profiles/h100_latency_model_pr2_v2.json")
    args = ap.parse_args()
    smoke = export_smoke(args.model_json)
    real = export_one_real(args.trace_dir, args.model_json)
    print(json.dumps({"smoke_nodes": len(smoke.get("nodes", [])), "real_nodes": len(real.get("nodes", [])) if real else 0}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

