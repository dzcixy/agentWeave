from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    rec: dict[str, float] = {"start": time.time(), "end": 0.0, "latency": 0.0}
    try:
        yield rec
    finally:
        rec["end"] = time.time()
        rec["latency"] = rec["end"] - rec["start"]
