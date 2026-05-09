from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import requests

from agentweaver.utils.io import write_csv

LOG = logging.getLogger(__name__)
EXPECTED = {
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries",
    "vllm:prompt_tokens",
    "vllm:prompt_tokens_cached",
    "vllm:generation_tokens",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
}


def parse_prometheus(text: str, timestamp: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"([A-Za-z_:][A-Za-z0-9_:]*)(\{([^}]*)\})?\s+([-+0-9.eE]+)", line)
        if not m:
            continue
        name = m.group(1)
        labels = m.group(3) or ""
        try:
            val = float(m.group(4))
        except ValueError:
            continue
        if name.startswith("vllm:") or name.startswith("vllm_"):
            seen.add(name)
            rows.append({"timestamp": timestamp, "metric_name": name, "labels": labels, "value": val})
    missing = [x for x in EXPECTED if x not in seen]
    if missing:
        LOG.warning("missing vLLM metrics in this scrape: %s", ", ".join(missing))
    return rows


def collect(metrics_url: str, out: str | Path, interval: float = 2.0, duration: float = 10.0) -> list[dict[str, object]]:
    end = time.time() + duration
    rows: list[dict[str, object]] = []
    while time.time() < end:
        ts = time.time()
        try:
            resp = requests.get(metrics_url, timeout=5)
            resp.raise_for_status()
            rows.extend(parse_prometheus(resp.text, ts))
        except Exception as exc:
            LOG.warning("metrics scrape failed: %s", exc)
        time.sleep(interval)
    write_csv(out, rows, ["timestamp", "metric_name", "labels", "value"])
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-url", default="http://localhost:8000/metrics")
    ap.add_argument("--out", default="data/profiles/vllm_metrics.csv")
    ap.add_argument("--interval", type=float, default=2)
    ap.add_argument("--duration", type=float, default=10)
    args = ap.parse_args()
    rows = collect(args.metrics_url, args.out, args.interval, args.duration)
    print(json.dumps({"rows": len(rows), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
