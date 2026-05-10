from __future__ import annotations

from agentweaver.profiling.prompt_factory import make_prompt_exact_tokens, make_shared_prefix_prompts


def test_exact_prompt_lengths() -> None:
    for target in [512, 4096, 8192]:
        _, actual = make_prompt_exact_tokens(target, seed=target)
        assert abs(actual - target) / target <= 0.02


def test_shared_prefix_exact_and_suffix_unique() -> None:
    rows = make_shared_prefix_prompts(1024, 256, 4, seed=7)
    prefix = rows[0]["shared_prefix_token_ids"]
    suffixes = []
    for row in rows:
        assert row["shared_prefix_token_ids"] == prefix
        assert row["token_ids"][: len(prefix)] == prefix
        suffixes.append(tuple(row["token_ids"][len(prefix) :]))
    assert len(set(suffixes)) == len(suffixes)
