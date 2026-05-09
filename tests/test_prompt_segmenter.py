from agentweaver.tracing.prompt_segmenter import segment_prompt


def test_prompt_segment_types_and_exact_prefix_ids() -> None:
    msgs = [
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "Available tools: shell grep pytest"},
        {"role": "user", "content": "SWE-bench issue: fix bug"},
        {"role": "user", "content": "Repository context:\n```python\ndef f(): pass\n```"},
        {"role": "assistant", "content": "branch reasoning"},
    ]
    a = segment_prompt(msgs)
    b = segment_prompt(msgs)
    assert {s.segment_type for s in a} >= {"system", "tool_schema", "task", "repo", "branch_suffix"}
    assert [s.segment_id for s in a] == [s.segment_id for s in b]
    assert all(s.start_pos >= 0 for s in a)
