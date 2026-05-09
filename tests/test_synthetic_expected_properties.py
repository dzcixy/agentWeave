from __future__ import annotations

import csv
from pathlib import Path

from agentweaver.analysis.context_segment_graph import process_trace_dir
from agentweaver.simulator.acd_mapping import run_mapping
from agentweaver.simulator.replay import replay
from agentweaver.workloads.synthetic_fork_join import FIXED_SCENARIOS, make_scenario_trace


def _scenario_processed(tmp_path: Path, name: str) -> Path:
    trace_dir = tmp_path / name / "traces"
    trace_dir.mkdir(parents=True)
    make_scenario_trace(FIXED_SCENARIOS[name]).to_jsonl(trace_dir / f"{name}.jsonl")
    processed = tmp_path / name / "processed"
    process_trace_dir(trace_dir, processed, "configs/default.yaml")
    return processed


def _agg(rows: list[dict[str, object]]) -> dict[str, object]:
    return rows[-1]


def test_s1_acd_reduces_hops_and_hotspot(tmp_path: Path) -> None:
    processed = _scenario_processed(tmp_path, "S1_context_heavy")
    result = run_mapping(processed, "configs/wafer_4x4.yaml", tmp_path / "acd.csv")
    assert result["avg_hops_after"] < result["avg_hops_before"]
    assert result["hotspot_after"] < result["hotspot_before"]


def test_s2_bes_and_safe_cancellation_direction(tmp_path: Path) -> None:
    processed = _scenario_processed(tmp_path, "S2_branch_heavy")
    static = _agg(replay(processed, "configs/wafer_4x4.yaml", "static_branch_pinning", tmp_path / "static.csv"))
    full = _agg(replay(processed, "configs/wafer_4x4.yaml", "full_agentweaver", tmp_path / "full.csv"))
    naive = _agg(replay(processed, "configs/wafer_4x4.yaml", "naive_wafer", tmp_path / "naive.csv"))
    assert float(full["region_utilization"]) > float(static["region_utilization"])
    assert int(full["branch_wasted_tokens"]) < int(naive["branch_wasted_tokens"])


def test_s3_nisp_reduces_resume_prefill(tmp_path: Path) -> None:
    processed = _scenario_processed(tmp_path, "S3_tool_stall_heavy")
    naive = _agg(replay(processed, "configs/wafer_4x4.yaml", "naive_wafer", tmp_path / "naive.csv"))
    nisp = _agg(replay(processed, "configs/wafer_4x4.yaml", "acd_nisp", tmp_path / "nisp.csv"))
    assert int(nisp["resume_prefill_tokens"]) < int(naive["resume_prefill_tokens"])


def test_full_not_worse_on_positive_scenarios(tmp_path: Path) -> None:
    for name in ["S1_context_heavy", "S2_branch_heavy", "S3_tool_stall_heavy"]:
        processed = _scenario_processed(tmp_path, name)
        naive = _agg(replay(processed, "configs/wafer_4x4.yaml", "naive_wafer", tmp_path / f"{name}_naive.csv"))
        full = _agg(replay(processed, "configs/wafer_4x4.yaml", "full_agentweaver", tmp_path / f"{name}_full.csv"))
        assert float(full["jct"]) <= float(naive["jct"])


def test_negative_controls_do_not_force_speedup(tmp_path: Path) -> None:
    low = _scenario_processed(tmp_path, "S4_low_reuse_negative")
    low_naive = _agg(replay(low, "configs/wafer_4x4.yaml", "naive_wafer", tmp_path / "low_naive.csv"))
    low_full = _agg(replay(low, "configs/wafer_4x4.yaml", "full_agentweaver", tmp_path / "low_full.csv"))
    low_benefit = (float(low_naive["jct"]) - float(low_full["jct"])) / max(1e-9, float(low_naive["jct"]))
    assert low_benefit < 0.20

    tool = _scenario_processed(tmp_path, "S5_tool_dominated_negative")
    s1 = _scenario_processed(tmp_path, "S1_context_heavy")
    tool_naive = _agg(replay(tool, "configs/wafer_4x4.yaml", "naive_wafer", tmp_path / "tool_naive.csv"))
    tool_full = _agg(replay(tool, "configs/wafer_4x4.yaml", "full_agentweaver", tmp_path / "tool_full.csv"))
    s1_naive = _agg(replay(s1, "configs/wafer_4x4.yaml", "naive_wafer", tmp_path / "s1_naive.csv"))
    s1_full = _agg(replay(s1, "configs/wafer_4x4.yaml", "full_agentweaver", tmp_path / "s1_full.csv"))
    tool_benefit = (float(tool_naive["jct"]) - float(tool_full["jct"])) / max(1e-9, float(tool_naive["jct"]))
    s1_benefit = (float(s1_naive["jct"]) - float(s1_full["jct"])) / max(1e-9, float(s1_naive["jct"]))
    assert tool_benefit < s1_benefit
