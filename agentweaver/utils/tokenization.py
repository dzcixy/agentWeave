from __future__ import annotations

import re
from functools import lru_cache
from typing import Any


class SimpleTokenizer:
    name_or_path = "simple-regex-tokenizer"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        toks = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        return [abs(hash(t)) % 50000 for t in toks]

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


@lru_cache(maxsize=8)
def get_tokenizer(model: str | None = None) -> Any:
    model = model or "Qwen/Qwen2.5-Coder-7B-Instruct"
    try:
        from transformers import AutoTokenizer  # type: ignore

        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception:
        return SimpleTokenizer()


def count_tokens(text: str, model: str | None = None) -> int:
    return len(get_tokenizer(model).encode(text or "", add_special_tokens=False))
