from pathlib import Path

from agentweaver.analysis.context_segment_graph import process_trace_dir
from agentweaver.simulator.acd_mapping import run_mapping
from agentweaver.workloads.synthetic_fork_join import make_synthetic_trace


def test_acd_places_shared_branches_close(tmp_path: Path) -> None:
    td = tmp_path / "traces"
    td.mkdir()
    make_synthetic_trace("i", branch_fanout=4, shared_prefix_len=1024).to_jsonl(td / "i.jsonl")
    processed = tmp_path / "processed"
    process_trace_dir(td, processed, "configs/default.yaml")
    out = tmp_path / "acd.csv"
    result = run_mapping(processed, "configs/wafer_4x4.yaml", out)
    regions = [tuple(v) for v in result["branch_to_region"].values()]
    max_dist = max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a in regions for b in regions)
    assert max_dist <= 4
    assert result["avg_hops"] >= 0
