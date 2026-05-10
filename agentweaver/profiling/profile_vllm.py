from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from agentweaver.profiling.async_vllm_client import AsyncVLLMClient, VLLMRequestRecord
from agentweaver.profiling.collect_vllm_metrics import diff_metrics, snapshot_metrics
from agentweaver.profiling.prompt_factory import (
    encode_text,
    load_tokenizer,
    make_prompt_exact_tokens,
    make_shared_prefix_prompts,
)
from agentweaver.utils.io import ensure_dir, write_csv, write_json


def _empty_metric_delta() -> dict[str, float]:
    keys = [
        "prefix_cache_queries_delta",
        "prefix_cache_hits_delta",
        "prompt_tokens_delta",
        "cached_prompt_tokens_delta",
        "generation_tokens_delta",
        "num_requests_running_delta",
        "num_requests_waiting_delta",
        "gpu_kv_cache_usage_delta",
        "request_queue_time_delta",
        "request_prefill_time_delta",
        "request_decode_time_delta",
        "ttft_delta",
        "tpot_delta",
        "e2e_latency_delta",
        "prefix_cache_metrics_reliable",
    ]
    return {k: "" for k in keys}


def _metric_snap(metrics_url: str | None, raw_metrics_out: str | None = None, raw_tag: str | None = None) -> dict[str, Any] | None:
    if not metrics_url:
        return None
    try:
        return snapshot_metrics(metrics_url, raw_out=raw_metrics_out, raw_tag=raw_tag)
    except Exception as exc:
        return {"error": str(exc), "values": {}, "missing": list(_empty_metric_delta())}


def _metric_diff(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    if before is None or after is None or before.get("error") or after.get("error"):
        out = _empty_metric_delta()
        out["metrics_error"] = before.get("error", "") if before else ""
        if after and after.get("error"):
            out["metrics_error"] = after["error"]
        return out
    out = diff_metrics(before, after)
    out["metrics_error"] = ""
    return out


def _rec_common(rec: VLLMRequestRecord) -> dict[str, Any]:
    return {
        "request_id": rec.request_id,
        "endpoint": rec.endpoint,
        "stream": rec.stream,
        "e2e_latency": rec.e2e_latency,
        "ttft": rec.ttft if rec.ttft is not None else "",
        "decode_time": rec.decode_time if rec.decode_time is not None else "",
        "tpot": rec.tpot if rec.tpot is not None else "",
        "client_prompt_tokens": rec.prompt_tokens_client,
        "prompt_tokens_client": rec.prompt_tokens_client,
        "completion_tokens_client": rec.completion_tokens_client,
        "client_completion_tokens": rec.completion_tokens_client,
        "server_prompt_tokens": rec.prompt_tokens_server if rec.prompt_tokens_server is not None else "",
        "prompt_tokens_server": rec.prompt_tokens_server if rec.prompt_tokens_server is not None else "",
        "server_completion_tokens": rec.completion_tokens_server if rec.completion_tokens_server is not None else "",
        "completion_tokens_server": rec.completion_tokens_server if rec.completion_tokens_server is not None else "",
        "output_tokens_actual": rec.completion_tokens_server or rec.completion_tokens_client,
        "start_time": rec.start_time,
        "first_token_time": rec.first_token_time if rec.first_token_time is not None else "",
        "end_time": rec.end_time,
        "status": rec.status,
        "error": rec.error,
    }


def _output_control_extra(target_output_tokens: int, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {}
    return {
        "ignore_eos": True,
        "min_tokens": int(target_output_tokens),
    }


def _profile_filename(args: argparse.Namespace, mode: str) -> str:
    if args.output_file:
        return args.output_file
    suffix = f"_{args.run_id}" if args.run_id else ""
    if mode == "sanity":
        return f"vllm_sanity{suffix}_raw.csv"
    return f"h100_profile_{mode}{suffix}.csv"


def _profile_path(args: argparse.Namespace, mode: str) -> Path:
    name = _profile_filename(args, mode)
    path = Path(name)
    return path if path.is_absolute() or path.parent != Path(".") else Path(args.out_dir) / path


@lru_cache(maxsize=16)
def _cached_body_prompt(body_target: int, tokenizer_path: str | None) -> str:
    return make_prompt_exact_tokens(body_target, seed=body_target, tokenizer_path=tokenizer_path)[0]


def _make_fast_unique_prompts(
    target_tokens: int,
    count: int,
    seed: int,
    tokenizer_path: str | None,
) -> list[str]:
    """Build tokenizer-checked unique prompts without rebuilding a long corpus per request.

    The unique nonce is placed at the beginning, so prefix-cache exact-prefix
    hits are not created by the shared body in the concurrency sweep.
    """
    tokenizer = load_tokenizer(tokenizer_path)
    body_target = max(128, target_tokens - 64)
    body = _cached_body_prompt(body_target, tokenizer_path)
    prompts: list[str] = []
    for i in range(count):
        header = (
            f"System nonce: concurrency-profile-{seed}-{i}.\n"
            f"Issue branch: request {i} inspects module_{(seed + i) % 997}.py and verifier_{(seed * 7 + i) % 541}.\n"
            "Tool log summary: pytest emitted a unique stack trace and this request must remain cache-independent.\n"
        )
        prompt = header + "\n" + body
        ids = encode_text(prompt, tokenizer)
        if len(ids) >= target_tokens:
            try:
                prompt = tokenizer.decode(ids[:target_tokens])
            except Exception:
                words = prompt.split()
                while len(ids) > target_tokens and words:
                    words = words[:-16]
                    prompt = " ".join(words)
                    ids = encode_text(prompt, tokenizer)
        ids = encode_text(prompt, tokenizer)
        guard = 0
        while len(ids) < int(target_tokens * 0.98) and guard < 50:
            guard += 1
            prompt += (
                f"\nAdditional unique evidence {seed}-{i}-{guard}: "
                f"src/pkg_{(seed + guard) % 311}/case_{i}.py failed with code {(seed + i + guard) % 17}."
            )
            ids = encode_text(prompt, tokenizer)
        rel = abs(len(ids) - target_tokens) / max(1, target_tokens)
        if rel > 0.02:
            raise ValueError(f"fast prompt token error {rel:.3%} target={target_tokens} actual={len(ids)}")
        prompts.append(prompt)
    return prompts


def _make_zero_shared_prompt_dicts(
    target_tokens: int,
    count: int,
    seed: int,
    tokenizer_path: str | None,
) -> list[dict[str, Any]]:
    tokenizer = load_tokenizer(tokenizer_path)
    rows: list[dict[str, Any]] = []
    for i in range(count):
        header = (
            f"zero_shared_unique_nonce_{seed}_{i}_{(seed * 131 + i) % 104729}\n"
            f"Request {i} must not share an exact prefix with any other request.\n"
        )
        header_tokens = len(encode_text(header, tokenizer))
        body_target = max(32, target_tokens - header_tokens)
        body, _ = make_prompt_exact_tokens(body_target, seed + i + 17, tokenizer_path)
        prompt = header + body
        ids = encode_text(prompt, tokenizer)
        if len(ids) > target_tokens:
            try:
                prompt = tokenizer.decode(ids[:target_tokens])
            except Exception:
                words = prompt.split()
                while len(ids) > target_tokens and words:
                    words = words[:-8]
                    prompt = " ".join(words)
                    ids = encode_text(prompt, tokenizer)
        ids = encode_text(prompt, tokenizer)
        guard = 0
        while len(ids) < int(target_tokens * 0.95) and guard < 20:
            guard += 1
            prompt += f"\nunique zero shared evidence {seed}-{i}-{guard}."
            ids = encode_text(prompt, tokenizer)
        rows.append(
            {
                "prompt": prompt,
                "total_tokens_actual": len(ids),
                "shared_prefix_tokens_actual": 0,
                "unique_suffix_tokens_actual": len(ids),
                "token_ids": [],
                "shared_prefix_token_ids": [],
            }
        )
    return rows


async def _profile_length(args: argparse.Namespace, client: AsyncVLLMClient) -> list[dict[str, Any]]:
    input_tokens = [512, 1024, 2048, 4096, 8192, 16384, 24576, 31488]
    output_tokens = [32, 128, 512, 1024]
    if args.mode == "sanity":
        input_tokens = [512, 2048]
        output_tokens = [32]
        args.repeats = 1
    rows: list[dict[str, Any]] = []
    for inp in input_tokens:
        prompt, actual = make_prompt_exact_tokens(inp, seed=inp, tokenizer_path=args.tokenizer_path)
        for out_tok in output_tokens:
            if actual + out_tok + args.max_model_len_reserve > args.max_model_len:
                continue
            for rep in range(args.repeats):
                raw_metrics = str(Path(args.out_dir) / f"vllm_metrics_raw_{args.run_id}_length.txt") if args.run_id else None
                before = _metric_snap(args.metrics_url, raw_metrics, f"length-{inp}-{out_tok}-{rep}-before")
                records = await client.run_concurrent(
                    [prompt],
                    out_tok,
                    concurrency=1,
                    stream=args.stream,
                    request_prefix=f"length-{inp}-{out_tok}-{rep}",
                    extra_body=_output_control_extra(out_tok, args.force_output_length),
                    endpoint=args.endpoint,
                    allow_stream_fallback=args.allow_stream_fallback,
                )
                after = _metric_snap(args.metrics_url, raw_metrics, f"length-{inp}-{out_tok}-{rep}-after")
                rec = records[0]
                row = {
                    "mode": "sanity" if args.mode == "sanity" else "length",
                    "run_id": args.run_id,
                    "input_tokens_target": inp,
                    "input_tokens_actual": actual,
                    "output_tokens_target": out_tok,
                    "output_length_control_requested": args.force_output_length,
                    "concurrency": 1,
                    "repeat": rep,
                    **_rec_common(rec),
                    **_metric_diff(before, after),
                }
                row["metrics_prompt_tokens_delta"] = row.get("prompt_tokens_delta", "")
                row["metrics_generation_tokens_delta"] = row.get("generation_tokens_delta", "")
                rows.append(row)
    return rows


async def _profile_concurrency(args: argparse.Namespace, client: AsyncVLLMClient) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for inp in [2048, 8192, 16384]:
        for out_tok in [128, 512]:
            for conc in [1, 2, 4, 8, 16]:
                for rep in range(args.repeats):
                    prompts = _make_fast_unique_prompts(
                        inp,
                        conc,
                        seed=inp * 100 + conc * 10 + rep,
                        tokenizer_path=args.tokenizer_path,
                    )
                    raw_metrics = str(Path(args.out_dir) / f"vllm_metrics_raw_{args.run_id}_concurrency.txt") if args.run_id else None
                    before = _metric_snap(args.metrics_url, raw_metrics, f"concurrency-{inp}-{out_tok}-{conc}-{rep}-before")
                    records = await client.run_concurrent(
                        prompts,
                        out_tok,
                        concurrency=conc,
                        stream=args.stream,
                        request_prefix=f"conc-{inp}-{out_tok}-{conc}-{rep}",
                        extra_body=_output_control_extra(out_tok, args.force_output_length),
                        endpoint=args.endpoint,
                        allow_stream_fallback=args.allow_stream_fallback,
                    )
                    after = _metric_snap(args.metrics_url, raw_metrics, f"concurrency-{inp}-{out_tok}-{conc}-{rep}-after")
                    ok = [r for r in records if r.status.startswith("ok")]
                    failed = [r for r in records if not r.status.startswith("ok")]
                    lat = [r.e2e_latency for r in ok]
                    ttft = [r.ttft for r in ok if r.ttft is not None]
                    dec = [r.decode_time for r in ok if r.decode_time is not None]
                    actual_out = [r.completion_tokens_server or r.completion_tokens_client for r in ok]
                    elapsed = max((r.end_time for r in records), default=time.time()) - min((r.start_time for r in records), default=time.time())
                    row = {
                        "mode": "concurrency",
                        "run_id": args.run_id,
                        "input_tokens_target": inp,
                        "input_tokens_actual": ok[0].prompt_tokens_client if ok else "",
                        "output_tokens_target": out_tok,
                        "output_tokens_actual_mean": statistics.mean(actual_out) if actual_out else "",
                        "output_length_control_requested": args.force_output_length,
                        "concurrency": conc,
                        "repeat": rep,
                        "stream": args.stream,
                        "per_request_latency_mean": statistics.mean(lat) if lat else "",
                        "per_request_latency_p50": statistics.median(lat) if lat else "",
                        "per_request_latency_p95": sorted(lat)[max(0, int(0.95 * (len(lat) - 1)))] if lat else "",
                        "per_request_ttft_mean": statistics.mean(ttft) if ttft else "",
                        "per_request_decode_time_mean": statistics.mean(dec) if dec else "",
                        "throughput_prompt_tokens_per_s": sum(r.prompt_tokens_client for r in ok) / max(1e-9, elapsed),
                        "throughput_generation_tokens_per_s": sum(r.completion_tokens_server or r.completion_tokens_client for r in ok) / max(1e-9, elapsed),
                        "queue_metric_delta": "",
                        "num_success": len(ok),
                        "num_failed": len(failed),
                        "status": "ok" if ok and not failed else ("partial" if ok else "error"),
                        "error": "; ".join(r.error for r in failed if r.error),
                        "case_start_time": min((r.start_time for r in records), default=""),
                        "case_end_time": max((r.end_time for r in records), default=""),
                    }
                    row.update(_metric_diff(before, after))
                    row["queue_metric_delta"] = row.get("request_queue_time_delta", "")
                    rows.append(row)
    return rows


async def _profile_prefix(args: argparse.Namespace, client: AsyncVLLMClient) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shared in [0, 1024, 4096, 8192, 16384]:
        for suffix in [256, 512, 1024]:
            for num in [2, 4, 8]:
                prompt_dicts: list[dict[str, Any]] | None = None
                prompts: list[str] = []
                exact_match = True
                if shared > 0:
                    prompt_dicts = make_shared_prefix_prompts(shared, suffix, num, seed=shared + suffix, tokenizer_path=args.tokenizer_path)
                    prompts = [p["prompt"] for p in prompt_dicts]
                    exact_match = all(p["token_ids"][: len(p["shared_prefix_token_ids"])] == p["shared_prefix_token_ids"] for p in prompt_dicts)
                    await client.run_concurrent(
                        prompts,
                        128,
                        concurrency=min(num, args.max_prefix_concurrency),
                        stream=False,
                        request_prefix=f"prefix-warm-{shared}-{suffix}-{num}",
                        endpoint=args.endpoint,
                    )
                for rep in range(args.repeats):
                    if shared == 0:
                        prompt_dicts = _make_zero_shared_prompt_dicts(
                            suffix,
                            num,
                            seed=9000 + rep * 1000 + suffix * 10 + num,
                            tokenizer_path=args.tokenizer_path,
                        )
                        prompts = [p["prompt"] for p in prompt_dicts]
                    raw_metrics = str(Path(args.out_dir) / f"vllm_metrics_raw_{args.run_id}_prefix.txt") if args.run_id else None
                    before = _metric_snap(args.metrics_url, raw_metrics, f"prefix-{shared}-{suffix}-{num}-{rep}-before")
                    records = await client.run_concurrent(
                        prompts,
                        128,
                        concurrency=min(num, args.max_prefix_concurrency),
                        stream=args.stream,
                        request_prefix=f"prefix-{shared}-{suffix}-{num}-{rep}",
                        extra_body=_output_control_extra(128, args.force_output_length),
                        endpoint=args.endpoint,
                        allow_stream_fallback=args.allow_stream_fallback,
                    )
                    after = _metric_snap(args.metrics_url, raw_metrics, f"prefix-{shared}-{suffix}-{num}-{rep}-after")
                    ok = [r for r in records if r.status.startswith("ok")]
                    failed = [r for r in records if not r.status.startswith("ok")]
                    lat = [r.e2e_latency for r in ok]
                    ttft = [r.ttft for r in ok if r.ttft is not None]
                    dec = [r.decode_time for r in ok if r.decode_time is not None]
                    row = {
                        "mode": "prefix",
                        "run_id": args.run_id,
                        "shared_prefix_tokens_target": shared,
                        "shared_prefix_tokens_actual": prompt_dicts[0]["shared_prefix_tokens_actual"],
                        "unique_suffix_tokens_target": suffix,
                        "unique_suffix_tokens_actual": prompt_dicts[0]["unique_suffix_tokens_actual"],
                        "num_requests": num,
                        "output_tokens": 128,
                        "repeat": rep,
                        "total_e2e_latency": max((r.end_time for r in records), default=0) - min((r.start_time for r in records), default=0),
                        "per_request_latency_mean": statistics.mean(lat) if lat else "",
                        "per_request_ttft_mean": statistics.mean(ttft) if ttft else "",
                        "per_request_decode_time_mean": statistics.mean(dec) if dec else "",
                        "prefix_token_ids_exact_match": exact_match,
                        "client_repeated_prefix_tokens": shared * max(0, num - 1),
                        "stream": args.stream,
                        "output_length_control_requested": args.force_output_length,
                        "num_success": len(ok),
                        "num_failed": len(failed),
                        "status": "ok" if ok and not failed else ("partial" if ok else "error"),
                        "error": "; ".join(r.error for r in failed if r.error),
                    }
                    row.update(_metric_diff(before, after))
                    row["prefix_cache_queries_delta"] = row.get("prefix_cache_queries_delta", "")
                    row["prefix_cache_hits_delta"] = row.get("prefix_cache_hits_delta", "")
                    row["cached_prompt_tokens_delta"] = row.get("cached_prompt_tokens_delta", "")
                    rows.append(row)
    return rows


async def profile_async(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = ensure_dir(args.out_dir)
    client = AsyncVLLMClient(args.server, args.model, args.tokenizer_path, timeout=args.request_timeout)
    if not await client.health():
        raise RuntimeError(f"vLLM server unavailable: {args.server}")
    outputs: dict[str, list[dict[str, Any]]] = {}
    if args.mode in {"sanity", "length", "all"}:
        rows = await _profile_length(args, client)
        write_csv(_profile_path(args, "sanity" if args.mode == "sanity" else "length"), rows)
        outputs["sanity" if args.mode == "sanity" else "length"] = rows
    if args.mode in {"concurrency", "all"}:
        rows = await _profile_concurrency(args, client)
        write_csv(_profile_path(args, "concurrency"), rows)
        outputs["concurrency"] = rows
    if args.mode in {"prefix", "all"}:
        rows = await _profile_prefix(args, client)
        write_csv(_profile_path(args, "prefix"), rows)
        outputs["prefix"] = rows
    if args.mode == "all":
        merged: list[dict[str, Any]] = []
        for rows in outputs.values():
            merged.extend(rows)
        suffix = f"_{args.run_id}" if args.run_id else ""
        write_csv(out_dir / f"h100_profile_raw{suffix}.csv", merged)
    return {k: {"rows": len(v), "ok": sum(1 for r in v if str(r.get("status", "")).startswith("ok"))} for k, v in outputs.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000/v1")
    ap.add_argument("--metrics-url", default=None)
    ap.add_argument("--model", default="qwen-coder-7b")
    ap.add_argument("--tokenizer-path", default=None)
    ap.add_argument("--mode", choices=["sanity", "length", "concurrency", "prefix", "all"], default="sanity")
    ap.add_argument("--out-dir", default="data/profiles")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--output-file", default="")
    ap.add_argument("--stream", dest="stream", action="store_true", default=True)
    ap.add_argument("--no-stream", dest="stream", action="store_false")
    ap.add_argument("--endpoint", choices=["chat", "completions"], default="chat")
    ap.add_argument("--force-output-length", dest="force_output_length", action="store_true", default=True)
    ap.add_argument("--no-force-output-length", dest="force_output_length", action="store_false")
    ap.add_argument("--allow-stream-fallback", action="store_true")
    ap.add_argument("--max-prefix-concurrency", type=int, default=8)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-model-len-reserve", type=int, default=256)
    args = ap.parse_args()
    try:
        summary = asyncio.run(profile_async(args))
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc), "server": args.server}, indent=2))
        raise SystemExit(1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
