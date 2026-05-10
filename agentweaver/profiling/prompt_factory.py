from __future__ import annotations

import argparse
import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

from agentweaver.utils.tokenization import SimpleTokenizer


DEFAULT_TOKENIZER_PATH = "/data2/model_zoo/Qwen2.5-7B-Instruct"


@lru_cache(maxsize=8)
def _load_tokenizer_cached(path: str) -> Any:
    try:
        from transformers import AutoTokenizer  # type: ignore

        if Path(path).exists():
            return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception:
        if path != DEFAULT_TOKENIZER_PATH:
            raise
    return SimpleTokenizer()


def load_tokenizer(tokenizer_path: str | None = None) -> Any:
    return _load_tokenizer_cached(tokenizer_path or DEFAULT_TOKENIZER_PATH)


def encode_text(text: str, tokenizer: Any | None = None, tokenizer_path: str | None = None) -> list[int]:
    tok = tokenizer or load_tokenizer(tokenizer_path)
    return list(tok.encode(text, add_special_tokens=False))


def _fragment(rng: random.Random, i: int) -> str:
    issue = [
        f"Issue note {i}: regression appears when ConfigLoader merges option_{rng.randrange(1000)}.",
        f"Expected behavior: test_{rng.randrange(500)} should preserve repo state after tool execution.",
        f"Observed failure: AssertionError in package/module_{rng.randrange(50)}.py with seed {rng.randrange(9999)}.",
    ]
    code = [
        "```python\n"
        f"def patch_candidate_{i}(repo, value_{rng.randrange(99)}):\n"
        f"    cache_key = f\"{{repo.name}}:{rng.randrange(999)}\"\n"
        "    if cache_key not in repo.memo:\n"
        "        repo.memo[cache_key] = sorted(repo.files)\n"
        "    return repo.memo[cache_key]\n```",
        "```diff\n"
        f"--- a/src/component_{rng.randrange(40)}.py\n"
        f"+++ b/src/component_{rng.randrange(40)}.py\n"
        f"@@ -{rng.randrange(1,20)},3 +{rng.randrange(1,20)},7 @@\n"
        "-    return old_value\n"
        "+    if value is None:\n"
        "+        return default_value\n"
        "+    return normalize(value)\n```",
    ]
    shell = [
        "```text\n"
        f"$ pytest tests/test_{rng.randrange(100)}.py -q\n"
        f"E   AssertionError: branch_{rng.randrange(16)} expected status pass\n"
        f"stderr: warning: retry budget {rng.randrange(10)} exhausted\n```",
        f"Search result: src/pkg_{rng.randrange(30)}/file_{rng.randrange(200)}.py: "
        f"class Handler{rng.randrange(1000)} processes event_{rng.randrange(1000)}.",
    ]
    prose = [
        f"The worker should inspect the failing path, update only the minimal branch, and rerun verifier {rng.randrange(100)}.",
        f"Repository summary: module alpha_{rng.randrange(100)} imports beta_{rng.randrange(100)} and writes artifact_{rng.randrange(100)}.",
    ]
    return rng.choice(issue + code + shell + prose)


def _base_prompt(seed: int) -> str:
    rng = random.Random(seed)
    parts = [
        "System: You are a precise software engineering agent. Use tools carefully.",
        "Available tools: shell, grep, sed, python, pytest, build. Return concise patches.",
        f"SWE-like issue: fix deterministic failure case {rng.randrange(1_000_000)} without unrelated refactors.",
    ]
    for i in range(256):
        parts.append(_fragment(rng, i))
    return "\n\n".join(parts)


def make_prompt_exact_tokens(target_tokens: int, seed: int, tokenizer_path: str | None = None) -> tuple[str, int]:
    tokenizer = load_tokenizer(tokenizer_path)
    rng = random.Random(seed)
    text = _base_prompt(seed)
    ids = encode_text(text, tokenizer)
    while len(ids) < target_tokens:
        text += "\n\n" + _fragment(rng, len(ids))
        ids = encode_text(text, tokenizer)
    if len(ids) > target_tokens:
        try:
            if isinstance(tokenizer, SimpleTokenizer):
                raise TypeError("fallback tokenizer is not reversible")
            text = tokenizer.decode(ids[:target_tokens])
        except Exception:
            # Fallback tokenizer has no reversible text model; trim by words and
            # refill with varied fragments.
            words = text.split()
            lo, hi = 0, len(words)
            while lo < hi:
                mid = (lo + hi) // 2
                if len(encode_text(" ".join(words[:mid]), tokenizer)) < target_tokens:
                    lo = mid + 1
                else:
                    hi = mid
            text = " ".join(words[:lo])
    ids = encode_text(text, tokenizer)
    guard = 0
    while abs(len(ids) - target_tokens) / max(1, target_tokens) > 0.02 and guard < 200:
        guard += 1
        if len(ids) < target_tokens:
            text += "\n" + _fragment(rng, guard)
        else:
            words = text.split()
            drop = max(1, min(32, len(words) // 100))
            text = " ".join(words[:-drop])
        ids = encode_text(text, tokenizer)
    actual = len(ids)
    rel = abs(actual - target_tokens) / max(1, target_tokens)
    if rel > 0.02:
        raise ValueError(f"prompt token error {rel:.3%} exceeds 2% target={target_tokens} actual={actual}")
    return text, actual


def make_shared_prefix_prompts(
    shared_prefix_tokens: int,
    unique_suffix_tokens: int,
    num_requests: int,
    seed: int,
    tokenizer_path: str | None = None,
) -> list[dict[str, Any]]:
    tokenizer = load_tokenizer(tokenizer_path)
    prefix_text, _ = make_prompt_exact_tokens(shared_prefix_tokens, seed, tokenizer_path)
    prefix_text = prefix_text.rstrip() + "\n\n### AGENTWEAVER_SHARED_PREFIX_BOUNDARY ###\n\n"
    prefix_ids = encode_text(prefix_text, tokenizer)
    rows: list[dict[str, Any]] = []
    suffix_ids_seen: set[tuple[int, ...]] = set()
    for i in range(num_requests):
        suffix_text, _ = make_prompt_exact_tokens(unique_suffix_tokens, seed + 10_000 + i, tokenizer_path)
        suffix_text = f"Branch request {i} unique suffix begins.\n" + suffix_text
        prompt = prefix_text + suffix_text
        token_ids = encode_text(prompt, tokenizer)
        if token_ids[: len(prefix_ids)] != prefix_ids:
            prompt = prefix_text + "\n" + suffix_text
            token_ids = encode_text(prompt, tokenizer)
        if token_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError("shared prefix token ids changed after suffix concatenation")
        suffix_token_ids = token_ids[len(prefix_ids) :]
        suffix_key = tuple(suffix_token_ids[: min(128, len(suffix_token_ids))])
        if suffix_key in suffix_ids_seen:
            raise ValueError("unique suffix collision")
        suffix_ids_seen.add(suffix_key)
        rows.append(
            {
                "prompt": prompt,
                "total_tokens_actual": len(token_ids),
                "shared_prefix_tokens_actual": len(prefix_ids),
                "unique_suffix_tokens_actual": len(suffix_token_ids),
                "token_ids": token_ids,
                "shared_prefix_token_ids": prefix_ids,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tokenizer-path")
    args = ap.parse_args()
    text, actual = make_prompt_exact_tokens(args.tokens, args.seed, args.tokenizer_path)
    print(json.dumps({"target": args.tokens, "actual": actual, "preview": text[:200]}, indent=2))


if __name__ == "__main__":
    main()
