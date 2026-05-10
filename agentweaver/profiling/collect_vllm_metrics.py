from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from agentweaver.utils.io import ensure_dir, write_csv, write_json

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricSpec:
    names: tuple[str, ...]
    allowed_types: tuple[str, ...] = ("counter", "gauge", "untyped", "unknown")
    exclude_histogram_parts: bool = True
    prefix_cache_metric: bool = False


LOGICAL_METRICS: dict[str, MetricSpec] = {
    "prefix_cache_queries": MetricSpec(
        (
            "vllm:prefix_cache_queries_total",
            "vllm_prefix_cache_queries_total",
            "vllm:prefix_cache_query_total",
            "vllm_prefix_cache_query_total",
        ),
        allowed_types=("counter", "gauge", "untyped", "unknown"),
        prefix_cache_metric=True,
    ),
    "prefix_cache_hits": MetricSpec(
        (
            "vllm:prefix_cache_hits_total",
            "vllm_prefix_cache_hits_total",
            "vllm:prefix_cache_hit_total",
            "vllm_prefix_cache_hit_total",
        ),
        allowed_types=("counter", "gauge", "untyped", "unknown"),
        prefix_cache_metric=True,
    ),
    "cached_prompt_tokens": MetricSpec(
        (
            "vllm:cached_prompt_tokens_total",
            "vllm_cached_prompt_tokens_total",
            "vllm:prompt_tokens_cached_total",
            "vllm_prompt_tokens_cached_total",
            "vllm:prefix_cache_hit_tokens_total",
            "vllm_prefix_cache_hit_tokens_total",
        ),
        allowed_types=("counter", "gauge", "untyped", "unknown"),
        prefix_cache_metric=True,
    ),
    "prompt_tokens": MetricSpec(
        (
            "vllm:prompt_tokens_total",
            "vllm_prompt_tokens_total",
            "vllm:prompt_tokens",
            "vllm_prompt_tokens",
        ),
    ),
    "generation_tokens": MetricSpec(
        (
            "vllm:generation_tokens_total",
            "vllm_generation_tokens_total",
            "vllm:decode_tokens_total",
            "vllm_decode_tokens_total",
            "vllm:output_tokens_total",
            "vllm_output_tokens_total",
        ),
    ),
    "num_requests_running": MetricSpec(("vllm:num_requests_running", "vllm_num_requests_running")),
    "num_requests_waiting": MetricSpec(("vllm:num_requests_waiting", "vllm_num_requests_waiting")),
    "gpu_kv_cache_usage": MetricSpec(
        (
            "vllm:gpu_cache_usage_perc",
            "vllm_gpu_cache_usage_perc",
            "vllm:gpu_kv_cache_usage_perc",
            "vllm_gpu_kv_cache_usage_perc",
        )
    ),
    "request_queue_time": MetricSpec(
        (
            "vllm:request_queue_time_seconds_sum",
            "vllm_request_queue_time_seconds_sum",
            "vllm:time_in_queue_requests_seconds_sum",
            "vllm_time_in_queue_requests_seconds_sum",
        ),
        allowed_types=("histogram", "summary", "counter", "gauge", "untyped", "unknown"),
        exclude_histogram_parts=False,
    ),
    "request_prefill_time": MetricSpec(
        (
            "vllm:request_prefill_time_seconds_sum",
            "vllm_request_prefill_time_seconds_sum",
            "vllm:prefill_time_seconds_sum",
            "vllm_prefill_time_seconds_sum",
        ),
        allowed_types=("histogram", "summary", "counter", "gauge", "untyped", "unknown"),
        exclude_histogram_parts=False,
    ),
    "request_decode_time": MetricSpec(
        (
            "vllm:request_decode_time_seconds_sum",
            "vllm_request_decode_time_seconds_sum",
            "vllm:decode_time_seconds_sum",
            "vllm_decode_time_seconds_sum",
        ),
        allowed_types=("histogram", "summary", "counter", "gauge", "untyped", "unknown"),
        exclude_histogram_parts=False,
    ),
    "ttft": MetricSpec(
        (
            "vllm:time_to_first_token_seconds_sum",
            "vllm_time_to_first_token_seconds_sum",
            "vllm:request_ttft_seconds_sum",
            "vllm_request_ttft_seconds_sum",
        ),
        allowed_types=("histogram", "summary", "counter", "gauge", "untyped", "unknown"),
        exclude_histogram_parts=False,
    ),
    "tpot": MetricSpec(
        (
            "vllm:time_per_output_token_seconds_sum",
            "vllm_time_per_output_token_seconds_sum",
            "vllm:request_tpot_seconds_sum",
            "vllm_request_tpot_seconds_sum",
        ),
        allowed_types=("histogram", "summary", "counter", "gauge", "untyped", "unknown"),
        exclude_histogram_parts=False,
    ),
    "e2e_latency": MetricSpec(
        (
            "vllm:e2e_request_latency_seconds_sum",
            "vllm_e2e_request_latency_seconds_sum",
            "vllm:request_latency_seconds_sum",
            "vllm_request_latency_seconds_sum",
            "vllm:request_duration_seconds_sum",
            "vllm_request_duration_seconds_sum",
        ),
        allowed_types=("histogram", "summary", "counter", "gauge", "untyped", "unknown"),
        exclude_histogram_parts=False,
    ),
}

HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count", "_created")


def _parse_labels(label_text: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not label_text:
        return labels
    for match in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"', label_text):
        labels[match.group(1)] = match.group(2).replace('\\"', '"')
    return labels


def _sample_kind(metric_name: str) -> str:
    for suffix in HISTOGRAM_SUFFIXES:
        if metric_name.endswith(suffix):
            return suffix[1:]
    return "sample"


def _type_lookup_name(metric_name: str) -> str:
    kind = _sample_kind(metric_name)
    if kind in {"bucket", "sum", "count", "created"}:
        return metric_name.rsplit("_", 1)[0]
    return metric_name


def parse_prometheus(text: str, timestamp: float | None = None) -> list[dict[str, Any]]:
    ts = time.time() if timestamp is None else timestamp
    type_by_name: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    type_re = re.compile(r"^#\s+TYPE\s+([A-Za-z_:][A-Za-z0-9_:]*)\s+([A-Za-z_][A-Za-z0-9_]*)")
    sample_re = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{([^}]*)\})?\s+([-+0-9.eE]+)(?:\s+\d+)?$")
    for line in text.splitlines():
        if not line:
            continue
        m_type = type_re.match(line)
        if m_type:
            type_by_name[m_type.group(1)] = m_type.group(2).lower()
            continue
        if line.startswith("#"):
            continue
        m = sample_re.match(line)
        if not m:
            continue
        try:
            value = float(m.group(4))
        except ValueError:
            continue
        metric_name = m.group(1)
        base_name = _type_lookup_name(metric_name)
        labels = _parse_labels(m.group(3) or "")
        rows.append(
            {
                "timestamp": ts,
                "metric_name": metric_name,
                "base_metric_name": base_name,
                "labels": json.dumps(labels, sort_keys=True),
                "label_names": ",".join(sorted(labels)),
                "value": value,
                "metric_type": type_by_name.get(base_name, type_by_name.get(metric_name, "unknown")),
                "sample_kind": _sample_kind(metric_name),
            }
        )
    return rows


def _row_matches(row: dict[str, Any], spec: MetricSpec) -> bool:
    name = str(row["metric_name"])
    metric_type = str(row.get("metric_type", "unknown"))
    if name not in spec.names:
        return False
    if metric_type not in spec.allowed_types:
        return False
    if spec.exclude_histogram_parts and str(row.get("sample_kind")) != "sample":
        return False
    if spec.prefix_cache_metric and str(row.get("sample_kind")) in {"bucket", "sum", "count", "created"}:
        return False
    return True


def summarize_metrics(rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]], list[str]]:
    values: dict[str, float] = {k: 0.0 for k in LOGICAL_METRICS}
    sources: dict[str, list[dict[str, Any]]] = {k: [] for k in LOGICAL_METRICS}
    seen: dict[str, set[tuple[str, str, str, str]]] = {k: set() for k in LOGICAL_METRICS}
    for row in rows:
        for logical, spec in LOGICAL_METRICS.items():
            if not _row_matches(row, spec):
                continue
            values[logical] += float(row["value"])
            key = (
                str(row["metric_name"]),
                str(row.get("labels", "")),
                str(row.get("metric_type", "")),
                str(row.get("sample_kind", "")),
            )
            if key not in seen[logical]:
                seen[logical].add(key)
                sources[logical].append(
                    {
                        "metric_name": row["metric_name"],
                        "labels": row.get("labels", ""),
                        "metric_type": row.get("metric_type", "unknown"),
                        "sample_kind": row.get("sample_kind", "sample"),
                    }
                )
    missing = [k for k, src in sources.items() if not src]
    return values, sources, missing


def prefix_cache_metrics_reliable(sources: dict[str, list[dict[str, Any]]]) -> bool:
    prefix_sources = []
    for key in ("prefix_cache_queries", "prefix_cache_hits", "cached_prompt_tokens"):
        prefix_sources.extend(sources.get(key, []))
    if not prefix_sources:
        return False
    for src in prefix_sources:
        name = str(src.get("metric_name", ""))
        if str(src.get("sample_kind", "")) != "sample":
            return False
        if not any(term in name for term in ("prefix_cache", "cached_prompt_tokens", "prompt_tokens_cached")):
            return False
    return True


def snapshot_metrics(metrics_url: str, raw_out: str | Path | None = None, raw_tag: str | None = None) -> dict[str, Any]:
    ts = time.time()
    session = requests.Session()
    session.trust_env = False
    resp = session.get(metrics_url, timeout=10)
    resp.raise_for_status()
    rows = parse_prometheus(resp.text, ts)
    values, sources, missing = summarize_metrics(rows)
    reliable = prefix_cache_metrics_reliable(sources)
    snap = {
        "timestamp": ts,
        "metrics_url": metrics_url,
        "rows": rows,
        "values": values,
        "sources": sources,
        "missing": missing,
        "prefix_cache_metrics_reliable": reliable,
        "raw_text": resp.text,
    }
    if raw_out:
        p = Path(raw_out)
        ensure_dir(p.parent)
        mode = "a" if p.exists() else "w"
        with p.open(mode, encoding="utf-8") as f:
            f.write(f"\n# AGENTWEAVER_SNAPSHOT tag={raw_tag or ''} timestamp={ts} url={metrics_url}\n")
            f.write(resp.text)
            if not resp.text.endswith("\n"):
                f.write("\n")
    if missing:
        LOG.warning("missing strict vLLM metric groups: %s", ", ".join(missing))
    return snap


def diff_metrics(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    reliable = bool(before.get("prefix_cache_metrics_reliable")) and bool(after.get("prefix_cache_metrics_reliable"))
    for key in LOGICAL_METRICS:
        if LOGICAL_METRICS[key].prefix_cache_metric and not reliable:
            out[key + "_delta"] = ""
        else:
            out[key + "_delta"] = float(after.get("values", {}).get(key, 0.0)) - float(before.get("values", {}).get(key, 0.0))
    out["prefix_cache_metrics_reliable"] = reliable
    return out


def _excluded_sources(rows: list[dict[str, Any]], spec: MetricSpec) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        name = str(row["metric_name"])
        if name in spec.names:
            continue
        base = _type_lookup_name(name)
        candidate_bases = {_type_lookup_name(n) for n in spec.names}
        if base not in candidate_bases:
            continue
        key = (
            name,
            str(row.get("labels", "")),
            str(row.get("metric_type", "")),
            str(row.get("sample_kind", "")),
            "not exact name or excluded histogram part",
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "metric_name": name,
                "labels": row.get("labels", ""),
                "metric_type": row.get("metric_type", "unknown"),
                "sample_kind": row.get("sample_kind", "sample"),
                "reason": "not exact name or excluded histogram part",
            }
        )
    return out


def write_source_audit(rows: list[dict[str, Any]], out: str | Path) -> dict[str, Any]:
    values, sources, missing = summarize_metrics(rows)
    reliable = prefix_cache_metrics_reliable(sources)
    lines = [
        "# vLLM Metrics Source Audit PR2-v2",
        "",
        f"rows_parsed = {len(rows)}",
        f"PREFIX_CACHE_METRICS_RELIABLE = {str(reliable).lower()}",
        "",
    ]
    audit: dict[str, Any] = {
        "rows_parsed": len(rows),
        "values": values,
        "sources": sources,
        "missing": missing,
        "prefix_cache_metrics_reliable": reliable,
    }
    for logical, spec in LOGICAL_METRICS.items():
        src = sources.get(logical, [])
        excluded = _excluded_sources(rows, spec)
        lines.append(f"## {logical}")
        lines.append("")
        lines.append(f"- exact_metric_names_allowed = {', '.join(spec.names)}")
        lines.append(f"- allowed_metric_types = {', '.join(spec.allowed_types)}")
        lines.append(f"- excludes_bucket_sum_count_for_token_counter = {str(spec.exclude_histogram_parts).lower()}")
        lines.append(f"- prefix_cache_metric = {str(spec.prefix_cache_metric).lower()}")
        lines.append(f"- matched_sources = {len(src)}")
        if src:
            for item in src[:20]:
                lines.append(
                    "- used "
                    f"name={item.get('metric_name')} "
                    f"type={item.get('metric_type')} "
                    f"sample_kind={item.get('sample_kind')} "
                    f"labels={item.get('labels') or '{}'}"
                )
        else:
            lines.append("- used none")
        if excluded:
            lines.append(f"- excluded_related_samples = {len(excluded)}")
            for item in excluded[:20]:
                lines.append(
                    "- excluded "
                    f"name={item.get('metric_name')} "
                    f"type={item.get('metric_type')} "
                    f"sample_kind={item.get('sample_kind')} "
                    f"labels={item.get('labels') or '{}'} "
                    f"reason={item.get('reason')}"
                )
        lines.append("")
    p = Path(out)
    ensure_dir(p.parent)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def audit_raw_files(raw_paths: list[str | Path], out: str | Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in raw_paths:
        p = Path(path)
        if not p.exists():
            continue
        rows.extend(parse_prometheus(p.read_text(encoding="utf-8", errors="replace")))
    return write_source_audit(rows, out)


def collect(
    metrics_url: str,
    out: str | Path,
    missing_out: str | Path | None = None,
    interval: float = 2.0,
    duration: float = 10.0,
    raw_out: str | Path | None = None,
    audit_out: str | Path | None = None,
) -> list[dict[str, Any]]:
    end = time.time() + duration
    all_rows: list[dict[str, Any]] = []
    missing_union: set[str] = set()
    sources: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    while time.time() < end:
        try:
            snap = snapshot_metrics(metrics_url, raw_out=raw_out, raw_tag="collect")
            all_rows.extend(snap["rows"])
            missing_union.update(snap["missing"])
            sources = snap["sources"]
        except Exception as exc:
            LOG.warning("metrics scrape failed: %s", exc)
            errors.append(str(exc))
            missing_union.update(LOGICAL_METRICS)
        time.sleep(interval)
    if not all_rows:
        missing_union.update(LOGICAL_METRICS)
    write_csv(out, all_rows, ["timestamp", "metric_name", "base_metric_name", "labels", "label_names", "value", "metric_type", "sample_kind"])
    if audit_out:
        write_source_audit(all_rows, audit_out)
    if missing_out:
        write_json(
            missing_out,
            {
                "metrics_url": metrics_url,
                "missing": sorted(missing_union),
                "available_sources": sources,
                "prefix_cache_metrics_reliable": prefix_cache_metrics_reliable(sources),
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
    ap.add_argument("--raw-out")
    ap.add_argument("--audit-out")
    ap.add_argument("--audit-raw", nargs="*")
    ap.add_argument("--interval", type=float, default=2)
    ap.add_argument("--duration", type=float, default=10)
    args = ap.parse_args()
    if args.audit_raw:
        audit = audit_raw_files(args.audit_raw, args.audit_out or "data/results/vllm_metrics_source_audit_pr2_v2.md")
        print(json.dumps({"rows": audit["rows_parsed"], "audit_out": args.audit_out}, indent=2))
        return
    rows = collect(args.metrics_url, args.out, args.missing_out, args.interval, args.duration, args.raw_out, args.audit_out)
    print(json.dumps({"rows": len(rows), "out": args.out, "missing_out": args.missing_out}, indent=2))


if __name__ == "__main__":
    main()
