from __future__ import annotations

import argparse
import csv
import json
import time
from itertools import product
from pathlib import Path

import requests

from agentweaver.utils.io import ensure_dir, write_csv


def synthetic_prompt(tokens: int) -> str:
    return " ".join(f"tok{i % 997}" for i in range(tokens))


def call_chat(server: str, model: str, prompt: str, max_tokens: int) -> dict[str, object]:
    url = server.rstrip("/") + "/chat/completions"
    t0 = time.time()
    resp = requests.post(
        url,
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0},
        timeout=600,
    )
    t1 = time.time()
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    return {
        "e2e_latency": t1 - t0,
        "prompt_tokens": usage.get("prompt_tokens", len(prompt.split())),
        "generated_tokens": usage.get("completion_tokens", max_tokens),
        "ttft": data.get("ttft", None),
    }


def profile(server: str, model: str, out: str | Path) -> list[dict[str, object]]:
    input_tokens = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    output_tokens = [32, 128, 512, 1024]
    batch_sizes = [1, 2, 4, 8]
    prefix_reuse = [0, 0.25, 0.5, 0.75, 1.0]
    rows: list[dict[str, object]] = []
    for inp, out_tok, bs, reuse, rep in product(input_tokens, output_tokens, batch_sizes, prefix_reuse, range(3)):
        prompts = [synthetic_prompt(inp) for _ in range(bs)]
        t0 = time.time()
        try:
            results = [call_chat(server, model, p, out_tok) for p in prompts]
            status = "ok"
        except Exception as exc:
            results = [{"e2e_latency": 0, "prompt_tokens": inp, "generated_tokens": 0, "ttft": None}]
            status = f"error:{exc}"
        e2e = time.time() - t0
        gen = sum(int(r.get("generated_tokens") or 0) for r in results)
        ttft = results[0].get("ttft")
        rows.append(
            {
                "input_tokens": inp,
                "output_tokens": out_tok,
                "batch_size": bs,
                "prefix_reuse": reuse,
                "repeat": rep,
                "e2e_latency": e2e,
                "ttft": ttft if ttft is not None else e2e / max(1, gen),
                "estimated_prefill": e2e * max(0.05, 1 - out_tok / max(1, inp + out_tok)),
                "estimated_decode": e2e * min(0.95, out_tok / max(1, inp + out_tok)),
                "prompt_tokens": inp * bs,
                "generated_tokens": gen,
                "status": status,
            }
        )
    write_csv(out, rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen-coder-7b")
    ap.add_argument("--out", default="data/profiles/h100_profile_raw.csv")
    args = ap.parse_args()
    print(json.dumps({"rows": len(profile(args.server, args.model, args.out)), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
