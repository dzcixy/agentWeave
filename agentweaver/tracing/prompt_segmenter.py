from __future__ import annotations

import argparse
import json
import re
from typing import Any

from agentweaver.tracing.trace_schema import ContextSegment, SegmentType
from agentweaver.utils.hashing import stable_hash
from agentweaver.utils.tokenization import get_tokenizer


def _classify(text: str, role: str | None = None) -> SegmentType:
    low = text.lower()
    if role == "system":
        return "system"
    if "available tools" in low or "tool schema" in low or "<tools>" in low:
        return "tool_schema"
    if "swe-bench" in low or "issue:" in low or "problem statement" in low:
        return "task"
    if "diff --git" in low or low.startswith("--- a/") or "\n+++" in low:
        return "patch"
    if "pytest" in low and ("failed" in low or "error" in low or "traceback" in low):
        return "test_log"
    if "```" in text or "class " in text or "def " in text or "grep" in low or "search result" in low:
        return "repo"
    if "stdout" in low or "stderr" in low or "exit code" in low:
        return "observation"
    if role == "assistant":
        return "branch_suffix"
    if role == "user":
        return "history"
    return "unknown"


def _privacy(seg_type: SegmentType) -> str:
    if seg_type in {"system", "tool_schema"}:
        return "public"
    if seg_type in {"branch_suffix", "patch", "scratchpad"}:
        return "branch_private"
    return "session_private"


def _split_rendered_prompt(prompt: str) -> list[dict[str, str]]:
    parts: list[dict[str, str]] = []
    chunks = re.split(r"(?=^#{2,}\s|\n<{1,3}[A-Z_a-z-]+>{0,3}|\n(?:System|User|Assistant):)", prompt, flags=re.M)
    for ch in chunks:
        s = ch.strip()
        if s:
            parts.append({"role": None, "content": s})
    return parts or [{"role": None, "content": prompt}]


def segment_prompt(
    raw_messages: list[dict[str, Any]] | None = None,
    rendered_prompt: str | None = None,
    tokenizer: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[ContextSegment]:
    metadata = metadata or {}
    model = metadata.get("model", "Qwen/Qwen2.5-Coder-7B-Instruct")
    tokenizer = tokenizer or get_tokenizer(model)
    tokenizer_id = getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
    messages = raw_messages if raw_messages is not None else _split_rendered_prompt(rendered_prompt or "")
    start = 0
    segments: list[ContextSegment] = []
    for msg in messages:
        role = msg.get("role")
        text = str(msg.get("content", ""))
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        seg_type = _classify(text, role)
        token_hash = stable_hash(token_ids, n=24)
        seg_id = stable_hash([model, tokenizer_id, token_ids, start, len(token_ids), seg_type], n=24)
        segments.append(
            ContextSegment(
                segment_id=seg_id,
                segment_type=seg_type,
                token_hash=token_hash,
                start_pos=start,
                length=len(token_ids),
                model=model,
                tokenizer=str(tokenizer_id),
                mutable=seg_type in {"history", "observation", "scratchpad", "branch_suffix", "patch", "test_log"},
                privacy=_privacy(seg_type),  # type: ignore[arg-type]
                exact_prefix_reusable=seg_type not in {"scratchpad", "branch_suffix", "patch"},
            )
        )
        start += len(token_ids)
    return segments


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    args = ap.parse_args()
    for seg in segment_prompt(rendered_prompt=args.prompt):
        print(json.dumps(seg.__dict__, sort_keys=True))


if __name__ == "__main__":
    main()
