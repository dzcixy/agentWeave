from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from agentweaver.profiling.prompt_factory import encode_text, load_tokenizer


@dataclass
class VLLMRequestRecord:
    request_id: str
    prompt_tokens_client: int
    completion_tokens_client: int
    prompt_tokens_server: int | None
    completion_tokens_server: int | None
    start_time: float
    first_token_time: float | None
    end_time: float
    e2e_latency: float
    ttft: float | None
    tpot: float | None
    status: str
    error: str = ""
    response_text: str = ""


class AsyncVLLMClient:
    def __init__(
        self,
        server: str,
        model: str,
        tokenizer_path: str | None = None,
        api_key: str = "EMPTY",
        timeout: float = 300.0,
    ):
        self.server = server.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.tokenizer = load_tokenizer(tokenizer_path)

    @property
    def chat_url(self) -> str:
        return self.server + "/chat/completions"

    async def health(self) -> bool:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            try:
                resp = await client.get(self.server.replace("/v1", "") + "/health")
                if resp.status_code < 500:
                    return True
            except Exception:
                pass
            try:
                resp = await client.get(self.server + "/models")
                return resp.status_code < 500
            except Exception:
                return False

    async def chat(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        stream: bool = False,
        temperature: float = 0.0,
    ) -> VLLMRequestRecord:
        if stream:
            try:
                return await self._chat_stream(request_id, prompt, max_tokens, temperature)
            except Exception as exc:
                rec = await self._chat_nonstream(request_id, prompt, max_tokens, temperature)
                if rec.status == "ok":
                    rec.status = "ok_stream_fallback"
                    rec.error = f"streaming failed; fallback non-streaming: {exc}"
                return rec
        return await self._chat_nonstream(request_id, prompt, max_tokens, temperature)

    async def _chat_nonstream(self, request_id: str, prompt: str, max_tokens: int, temperature: float) -> VLLMRequestRecord:
        prompt_tokens = len(encode_text(prompt, self.tokenizer))
        start = time.time()
        first = None
        end = start
        text = ""
        usage: dict[str, Any] = {}
        status = "ok"
        error = ""
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                resp = await client.post(
                    self.chat_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": False,
                    },
                )
                end = time.time()
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {}) or {}
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        except Exception as exc:
            end = time.time()
            status = "error"
            error = str(exc)
        completion_tokens = len(encode_text(text, self.tokenizer)) if text else 0
        e2e = end - start
        tpot = e2e / max(1, completion_tokens)
        return VLLMRequestRecord(
            request_id=request_id,
            prompt_tokens_client=prompt_tokens,
            completion_tokens_client=completion_tokens,
            prompt_tokens_server=usage.get("prompt_tokens"),
            completion_tokens_server=usage.get("completion_tokens"),
            start_time=start,
            first_token_time=first,
            end_time=end,
            e2e_latency=e2e,
            ttft=None,
            tpot=tpot,
            status=status,
            error=error,
            response_text=text,
        )

    async def _chat_stream(self, request_id: str, prompt: str, max_tokens: int, temperature: float) -> VLLMRequestRecord:
        prompt_tokens = len(encode_text(prompt, self.tokenizer))
        start = time.time()
        first: float | None = None
        text_parts: list[str] = []
        usage: dict[str, Any] = {}
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            async with client.stream(
                "POST",
                self.chat_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if data.get("usage"):
                        usage = data["usage"]
                    choices = data.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {}).get("content", "")
                        if delta:
                            if first is None:
                                first = time.time()
                            text_parts.append(delta)
        end = time.time()
        text = "".join(text_parts)
        completion_tokens = len(encode_text(text, self.tokenizer)) if text else int(usage.get("completion_tokens") or 0)
        ttft = (first - start) if first is not None else None
        decode_time = max(0.0, end - (first or start))
        tpot = decode_time / max(1, completion_tokens)
        return VLLMRequestRecord(
            request_id=request_id,
            prompt_tokens_client=prompt_tokens,
            completion_tokens_client=completion_tokens,
            prompt_tokens_server=usage.get("prompt_tokens"),
            completion_tokens_server=usage.get("completion_tokens"),
            start_time=start,
            first_token_time=first,
            end_time=end,
            e2e_latency=end - start,
            ttft=ttft,
            tpot=tpot,
            status="ok",
            response_text=text,
        )

    async def run_concurrent(
        self,
        prompts: list[str],
        max_tokens: int,
        concurrency: int,
        stream: bool = False,
        request_prefix: str = "req",
    ) -> list[VLLMRequestRecord]:
        semaphore = asyncio.Semaphore(concurrency)

        async def one(i: int, prompt: str) -> VLLMRequestRecord:
            async with semaphore:
                return await self.chat(f"{request_prefix}-{i}", prompt, max_tokens, stream=stream)

        return await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))


def records_to_rows(records: list[VLLMRequestRecord], include_text: bool = False) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        row = asdict(rec)
        if not include_text:
            row.pop("response_text", None)
        rows.append(row)
    return rows


async def _amain() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--tokenizer-path")
    ap.add_argument("--prompt", default="Say ok.")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()
    client = AsyncVLLMClient(args.server, args.model, args.tokenizer_path)
    records = await client.run_concurrent([args.prompt] * args.concurrency, args.max_tokens, args.concurrency, args.stream)
    print(json.dumps(records_to_rows(records), indent=2))


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
