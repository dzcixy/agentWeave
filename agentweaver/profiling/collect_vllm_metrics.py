from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import requests

from agentweaver.utils.io import ensure_dir, write_csv, write_json

LOG = logging.getLogger(__name__)

FUZZY_METRICS: dict[str, list[str]] = {
    "prefix_cache_queries": ["prefix_cache_queries", "prefix_cache_query", "prefix_queries"],
    "prefix_cache_hits": ["prefix_cache_hits", "prefix_cache_hit", "prefix_hits"],
    "prompt_tokens": ["prompt_tokens_total", "prompt_tokens", "prompt_token"],
    "cached_prompt_tokens": ["prompt_tokens_cached", "cached_prompt_tokens", "prefix_cache_hit_tokens"],
    "generation_tokens": ["generation_tokens_total", "generation_tokens", "decode_tokens"],
    "num_requests_running": ["num_requests_running", "requests_running"],
    "num_requests_waiting": ["num_requests_waiting", "requests_waiting", "waiting_requests"],
    "gpu_kv_cache_usage": ["gpu_cache_usage_perc", "kv_cache_usage", "gpu_kv_cache_usage"],
    "request_queue_time": ["request_queue", "queue_time", "time_in_queue"],
    "request_prefill_time": ["prefill_time", "prompt_time"],
    "request_decode_time": ["decode_time", "generation_time"],
    "ttft": ["time_to_first_token", "ttft"],
    "tpot": ["time_per_output_token", "tpot"],
    "e2e_latency": ["e2e", "request_latency", "request_duration"],
}


def parse_prometheus(text: str, timestamp: float | None = None) -> list[dict[str, Any]]:
    ts = time.time() if timestamp is None else timestamp
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"([A-Za-z_:][A-Za-z0-9_:]*)(\{([^}]*)\})?\s+([-+0-9.eE]+)", line)
        if not m:
            continue
        try:
            value = float(m.group(4))
        except ValueError:
            continue
        rows.append(
            {
                "timestamp": ts,
                "metric_name": m.group(1),
                "labels": m.group(3) or "",
                "value": value,
            }
        )
    return rows


def _logical_match(metric_name: str) -> list[str]:
    low = metric_name.lower().replace(":", "_")
    matched = []
    for logical, needles in FUZZY_METRICS.items():
        if any(n in low for n in needles):
            matched.append(logical)
    return matched


def summarize_metrics(rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, list[str]], list[str]]:
    values: dict[str, float] = {k: 0.0 for k in FUZZY_METRICS}
    sources: dict[str, list[str]] = {k: [] for k in FUZZY_METRICS}
    for row in rows:
        for logical in _logical_match(str(row["metric_name"])):
            values[logical] += float(row["value"])
            if row["metric_name"] not in sources[logical]:
                sources[logical].append(row["metric_name"])
    missing = [k for k, src in sources.items() if not src]
    return values, sources, missing


def snapshot_metrics(metrics_url: str, raw_out: str | Path | None = None) -> dict[str, Any]:
    ts = time.time()
    session = requests.Session()
    session.trust_env = False
    resp = session.get(metrics_url, timeout=10)
    resp.raise_for_status()
    rows = parse_prometheus(resp.text, ts)
    values, sources, missing = summarize_metrics(rows)
    snap = {
        "timestamp": ts,
        "metrics_url": metrics_url,
        "rows": rows,
        "values": values,
        "sources": sources,
        "missing": missing,
        "raw_text": resp.text,
    }
    if raw_out:
        p = Path(raw_out)
        ensure_dir(p.parent)
        p.write_text(resp.text, encoding="utf-8")
    if missing:
        LOG.warning("missing vLLM metric groups: %s", ", ".join(missing))
    return snap


def diff_metrics(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in FUZZY_METRICS:
        out[key + "_delta"] = float(after.get("values", {}).get(key, 0.0)) - float(before.get("values", {}).get(key, 0.0))
    return out


def collect(
    metrics_url: str,
    out: str | Path,
    missing_out: str | Path | None = None,
    interval: float = 2.0,
    duration: float = 10.0,
) -> list[dict[str, Any]]:
    end = time.time() + duration
    all_rows: list[dict[str, Any]] = []
    missing_union: set[str] = set()
    sources: dict[str, list[str]] = {}
    errors: list[str] = []
    while time.time() < end:
        try:
            snap = snapshot_metrics(metrics_url)
            all_rows.extend(snap["rows"])
            missing_union.update(snap["missing"])
            sources = snap["sources"]
        except Exception as exc:
            LOG.warning("metrics scrape failed: %s", exc)
            errors.append(str(exc))
            missing_union.update(FUZZY_METRICS)
        time.sleep(interval)
    if not all_rows:
        missing_union.update(FUZZY_METRICS)
    write_csv(out, all_rows, ["timestamp", "metric_name", "labels", "value"])
    if missing_out:
        write_json(
            missing_out,
            {
                "metrics_url": metrics_url,
                "missing": sorted(missing_union),
                "available_sources": sources,
                "rows": len(all_rows),
                "errors": errors,
            },
        )
    return all_rows


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-url", default="http://localhost:8000/metrics")
    ap.add_argument("--out", default="data/profiles/vllm_metrics_profile.csv")
    ap.add_argument("--missing-out", default="data/profiles/vllm_metrics_missing.json")
    ap.add_argument("--interval", type=float, default=2)
    ap.add_argument("--duration", type=float, default=10)
    args = ap.parse_args()
    rows = collect(args.metrics_url, args.out, args.missing_out, args.interval, args.duration)
    print(json.dumps({"rows": len(rows), "out": args.out, "missing_out": args.missing_out}, indent=2))


if __name__ == "__main__":
    main()
