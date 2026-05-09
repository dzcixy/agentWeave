from pathlib import Path

from agentweaver.analysis.context_segment_graph import process_trace_dir
from agentweaver.simulator.replay import replay
from agentweaver.workloads.synthetic_fork_join import make_synthetic_trace


def test_replay_full_reduces_prefill_or_hops(tmp_path: Path) -> None:
    td = tmp_path / "traces"
    td.mkdir()
    make_synthetic_trace("i", branch_fanout=4).to_jsonl(td / "i.jsonl")
    processed = tmp_path / "processed"
    process_trace_dir(td, processed, "configs/default.yaml")
    naive = replay(processed, "configs/wafer_4x4.yaml", "naive_wafer", tmp_path / "naive.csv")[-1]
    full = replay(processed, "configs/wafer_4x4.yaml", "full_agentweaver", tmp_path / "full.csv")[-1]
    assert int(full["prefill_tokens_avoided"]) >= int(naive["prefill_tokens_avoided"])
    assert float(full["jct"]) <= float(naive["jct"]) * 1.2
