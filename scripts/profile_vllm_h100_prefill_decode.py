#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _split_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _server_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _gpu_info() -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            text=True,
            capture_output=True,
            timeout=5,
        )
        names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        return len(names), names[0] if names else "unknown"
    except Exception:
        return 0, "unknown"


def _post_completion(host: str, port: int, model: str, prompts: list[str], output_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompts,
        "max_tokens": output_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _prompt(tokens: int) -> str:
    return " ".join(["token"] * max(1, tokens))


def run_profile(args: argparse.Namespace) -> list[dict[str, Any]]:
    proc: subprocess.Popen[str] | None = None
    if not _server_reachable(args.host, args.port):
        if args.launch_vllm and args.model:
            proc = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    args.model,
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                ],
                text=True,
            )
            deadline = time.time() + args.launch_timeout
            while time.time() < deadline and not _server_reachable(args.host, args.port):
                time.sleep(2)
        if not _server_reachable(args.host, args.port):
            return [
                {
                    "request_id": "NOT_RUN",
                    "model": args.model or "",
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "batch_size": 0,
                    "ttft": "",
                    "total_latency": "",
                    "prefill_latency_est": "",
                    "decode_latency_est": "",
                    "tokens_per_sec": "",
                    "gpu_count": 0,
                    "gpu_name": "",
                    "status": "VLLM_UNAVAILABLE",
                }
            ]
    gpu_count, gpu_name = _gpu_info()
    rows: list[dict[str, Any]] = []
    try:
        for prompt_tokens in args.prompt_token_lengths:
            prompt = _prompt(prompt_tokens)
            for output_tokens in args.decode_token_lengths:
                for batch_size in args.batch_sizes:
                    prompts = [prompt] * batch_size
                    for repeat in range(args.repeats):
                        req_id = f"p{prompt_tokens}_o{output_tokens}_b{batch_size}_r{repeat}"
                        start = time.perf_counter()
                        try:
                            result = _post_completion(args.host, args.port, args.model, prompts, output_tokens)
                            total = time.perf_counter() - start
                            usage = result.get("usage", {})
                            completion_tokens = int(usage.get("completion_tokens", output_tokens * batch_size) or 0)
                            prefill_est = max(0.0, total - completion_tokens * args.decode_token_estimate)
                            decode_est = max(0.0, total - prefill_est)
                            rows.append(
                                {
                                    "request_id": req_id,
                                    "model": args.model,
                                    "prompt_tokens": prompt_tokens,
                                    "output_tokens": output_tokens,
                                    "batch_size": batch_size,
                                    "ttft": "",
                                    "total_latency": total,
                                    "prefill_latency_est": prefill_est,
                                    "decode_latency_est": decode_est,
                                    "tokens_per_sec": completion_tokens / max(1e-9, total),
                                    "gpu_count": gpu_count,
                                    "gpu_name": gpu_name,
                                    "status": "OK",
                                }
                            )
                        except (urllib.error.URLError, TimeoutError, OSError) as exc:
                            rows.append(
                                {
                                    "request_id": req_id,
                                    "model": args.model,
                                    "prompt_tokens": prompt_tokens,
                                    "output_tokens": output_tokens,
                                    "batch_size": batch_size,
                                    "ttft": "",
                                    "total_latency": "",
                                    "prefill_latency_est": "",
                                    "decode_latency_est": "",
                                    "tokens_per_sec": "",
                                    "gpu_count": gpu_count,
                                    "gpu_name": gpu_name,
                                    "status": f"ERROR:{exc}",
                                }
                            )
    finally:
        if proc and args.stop_launched_server:
            proc.terminate()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", ""))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt-token-lengths", type=_split_ints, default=_split_ints("512,1024,2048,4096,8192,16384"))
    ap.add_argument("--decode-token-lengths", type=_split_ints, default=_split_ints("16,64,128,256"))
    ap.add_argument("--batch-sizes", type=_split_ints, default=_split_ints("1,2,4,8"))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--launch-vllm", action="store_true")
    ap.add_argument("--launch-timeout", type=float, default=180.0)
    ap.add_argument("--stop-launched-server", action="store_true")
    ap.add_argument("--decode-token-estimate", type=float, default=0.002)
    ap.add_argument("--out", default="data/calibration/h100_vllm_latency_raw_pr4_v13.csv")
    args = ap.parse_args()
    rows = run_profile(args)
    _ensure_parent(args.out)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(args.out).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "out": args.out, "status": rows[0].get("status", "") if rows else "EMPTY"}, indent=2))


if __name__ == "__main__":
    main()
