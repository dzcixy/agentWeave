from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(obj: Any, n: int = 16) -> str:
    """Deterministic short SHA256 for JSON-like objects or bytes."""
    if isinstance(obj, bytes):
        data = obj
    elif isinstance(obj, str):
        data = obj.encode("utf-8")
    else:
        data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:n]


def prompt_hash(text: str) -> str:
    return stable_hash(text, n=24)
