from agentweaver.tracing.dag_builder import analyze_dag, build_agent_dag
from agentweaver.workloads.synthetic_fork_join import make_synthetic_trace


def test_synthetic_2_branch_dag() -> None:
    tr = make_synthetic_trace("i", branch_fanout=2, success_branch="first")
    g = build_agent_dag(tr.events)
    assert g.number_of_nodes() > 0
    summary = analyze_dag(tr.events)
    assert summary["branch_fanout"] == 2
    assert summary["first_success_branch"] == "b0"
    assert summary["critical_path_length"] > 0
