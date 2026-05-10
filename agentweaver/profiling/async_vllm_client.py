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
    endpoint: str
    stream: bool
    prompt_tokens_client: int
    completion_tokens_client: int
    prompt_tokens_server: int | None
    completion_tokens_server: int | None
    start_time: float
    first_token_time: float | None
    end_time: float
    e2e_latency: float
    ttft: float | None
    decode_time: float | None
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

    @property
    def completions_url(self) -> str:
        return self.server + "/completions"

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
        extra_body: dict[str, Any] | None = None,
        endpoint: str = "chat",
        fallback_to_completion: bool = True,
        allow_stream_fallback: bool = False,
    ) -> VLLMRequestRecord:
        """Issue one OpenAI-compatible generation request.

        ``extra_body`` is sent as top-level JSON fields because vLLM exposes
        OpenAI-compatible extensions such as ``ignore_eos`` and ``min_tokens``
        that are not part of the public OpenAI schema.
        """
        if endpoint not in {"chat", "completions"}:
            raise ValueError(f"unknown endpoint {endpoint}; expected chat or completions")
        extra_body = dict(extra_body or {})
        if stream:
            rec = await self._stream(request_id, prompt, max_tokens, temperature, extra_body, endpoint)
            if rec.status == "ok":
                return rec
            if endpoint == "chat" and fallback_to_completion:
                rec2 = await self._stream(
                    request_id,
                    prompt,
                    max_tokens,
                    temperature,
                    extra_body,
                    "completions",
                    status_prefix="ok_endpoint_fallback",
                    previous_error=rec.error,
                )
                if rec2.status.startswith("ok"):
                    return rec2
            if extra_body:
                rec3 = await self._stream(
                    request_id,
                    prompt,
                    max_tokens,
                    temperature,
                    {},
                    endpoint,
                    status_prefix="ok_uncontrolled_output",
                    previous_error=rec.error,
                )
                if rec3.status.startswith("ok"):
                    return rec3
            if allow_stream_fallback:
                rec4 = await self._nonstream(
                    request_id,
                    prompt,
                    max_tokens,
                    temperature,
                    extra_body,
                    endpoint,
                    status_prefix="warning_stream_fallback",
                    previous_error=rec.error,
                )
                return rec4
            return rec
        rec = await self._nonstream(request_id, prompt, max_tokens, temperature, extra_body, endpoint)
        if rec.status == "ok":
            return rec
        if endpoint == "chat" and fallback_to_completion:
            rec2 = await self._nonstream(
                request_id,
                prompt,
                max_tokens,
                temperature,
                extra_body,
                "completions",
                status_prefix="ok_endpoint_fallback",
                previous_error=rec.error,
            )
            if rec2.status.startswith("ok"):
                return rec2
        if extra_body:
            rec3 = await self._nonstream(
                request_id,
                prompt,
                max_tokens,
                temperature,
                {},
                endpoint,
                status_prefix="ok_uncontrolled_output",
                previous_error=rec.error,
            )
            if rec3.status.startswith("ok"):
                return rec3
        return rec

    def _payload(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stream: bool,
        endpoint: str,
        extra_body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if endpoint == "chat":
            payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            payload["prompt"] = prompt
        if stream:
            payload["stream_options"] = {"include_usage": True}
        payload.update(extra_body or {})
        return payload

    def _error_record(
        self,
        request_id: str,
        prompt: str,
        start: float,
        end: float,
        stream: bool,
        endpoint: str,
        error: str,
    ) -> VLLMRequestRecord:
        prompt_tokens = len(encode_text(prompt, self.tokenizer))
        return VLLMRequestRecord(
            request_id=request_id,
            endpoint=endpoint,
            stream=stream,
            prompt_tokens_client=prompt_tokens,
            completion_tokens_client=0,
            prompt_tokens_server=None,
            completion_tokens_server=None,
            start_time=start,
            first_token_time=None,
            end_time=end,
            e2e_latency=end - start,
            ttft=None,
            decode_time=None,
            tpot=None,
            status="error",
            error=error,
            response_text="",
        )

    async def _nonstream(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        extra_body: dict[str, Any] | None,
        endpoint: str,
        status_prefix: str = "ok",
        previous_error: str = "",
    ) -> VLLMRequestRecord:
        prompt_tokens = len(encode_text(prompt, self.tokenizer))
        start = time.time()
        first = None
        end = start
        text = ""
        usage: dict[str, Any] = {}
        status = "ok"
        error = ""
        url = self.chat_url if endpoint == "chat" else self.completions_url
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=self._payload(prompt, max_tokens, temperature, False, endpoint, extra_body),
                )
                end = time.time()
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {}) or {}
                choice = (data.get("choices") or [{}])[0]
                if endpoint == "chat":
                    text = choice.get("message", {}).get("content", "") or ""
                else:
                    text = choice.get("text", "") or ""
        except Exception as exc:
            end = time.time()
            status = "error"
            error = str(exc)
        if status == "error":
            return self._error_record(request_id, prompt, start, end, False, endpoint, error)
        completion_tokens = len(encode_text(text, self.tokenizer)) if text else 0
        e2e = end - start
        decode_time = None
        tpot = None
        if completion_tokens > 0:
            decode_time = e2e
            tpot = decode_time / completion_tokens
        if status_prefix != "ok":
            status = status_prefix
            error = previous_error
        return VLLMRequestRecord(
            request_id=request_id,
            endpoint=endpoint,
            stream=False,
            prompt_tokens_client=prompt_tokens,
            completion_tokens_client=completion_tokens,
            prompt_tokens_server=usage.get("prompt_tokens"),
            completion_tokens_server=usage.get("completion_tokens"),
            start_time=start,
            first_token_time=first,
            end_time=end,
            e2e_latency=e2e,
            ttft=None,
            decode_time=decode_time,
            tpot=tpot,
            status=status,
            error=error,
            response_text=text,
        )

    async def _stream(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        extra_body: dict[str, Any] | None,
        endpoint: str,
        status_prefix: str = "ok",
        previous_error: str = "",
    ) -> VLLMRequestRecord:
        prompt_tokens = len(encode_text(prompt, self.tokenizer))
        start = time.time()
        first: float | None = None
        text_parts: list[str] = []
        usage: dict[str, Any] = {}
        url = self.chat_url if endpoint == "chat" else self.completions_url
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                async with client.stream(
                    "POST",
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=self._payload(prompt, max_tokens, temperature, True, endpoint, extra_body),
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
                        if not choices:
                            continue
                        choice = choices[0]
                        if endpoint == "chat":
                            delta = choice.get("delta", {}).get("content", "")
                        else:
                            delta = choice.get("text", "")
                        if delta:
                            if first is None:
                                first = time.time()
                            text_parts.append(delta)
        except Exception as exc:
            end = time.time()
            return self._error_record(request_id, prompt, start, end, True, endpoint, str(exc))
        end = time.time()
        text = "".join(text_parts)
        completion_tokens = len(encode_text(text, self.tokenizer)) if text else int(usage.get("completion_tokens") or 0)
        ttft = (first - start) if first is not None else None
        decode_time = max(0.0, end - first) if first is not None else None
        tpot = (decode_time / completion_tokens) if decode_time is not None and completion_tokens > 0 else None
        status = status_prefix
        error = previous_error if status_prefix != "ok" else ""
        return VLLMRequestRecord(
            request_id=request_id,
            endpoint=endpoint,
            stream=True,
            prompt_tokens_client=prompt_tokens,
            completion_tokens_client=completion_tokens,
            prompt_tokens_server=usage.get("prompt_tokens"),
            completion_tokens_server=usage.get("completion_tokens"),
            start_time=start,
            first_token_time=first,
            end_time=end,
            e2e_latency=end - start,
            ttft=ttft,
            decode_time=decode_time,
            tpot=tpot,
            status=status,
            error=error,
            response_text=text,
        )

    async def run_concurrent(
        self,
        prompts: list[str],
        max_tokens: int,
        concurrency: int,
        stream: bool = False,
        request_prefix: str = "req",
        temperature: float = 0.0,
        extra_body: dict[str, Any] | None = None,
        endpoint: str = "chat",
        fallback_to_completion: bool = True,
        allow_stream_fallback: bool = False,
    ) -> list[VLLMRequestRecord]:
        semaphore = asyncio.Semaphore(concurrency)

        async def one(i: int, prompt: str) -> VLLMRequestRecord:
            async with semaphore:
                return await self.chat(
                    f"{request_prefix}-{i}",
                    prompt,
                    max_tokens,
                    stream=stream,
                    temperature=temperature,
                    extra_body=extra_body,
                    endpoint=endpoint,
                    fallback_to_completion=fallback_to_completion,
                    allow_stream_fallback=allow_stream_fallback,
                )

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
    ap.add_argument("--model", default="qwen-coder-7b")
    ap.add_argument("--tokenizer-path")
    ap.add_argument("--prompt", default="Say ok.")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--endpoint", choices=["chat", "completions"], default="chat")
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--min-tokens", type=int)
    args = ap.parse_args()
    client = AsyncVLLMClient(args.server, args.model, args.tokenizer_path)
    extra_body = {}
    if args.ignore_eos:
        extra_body["ignore_eos"] = True
    if args.min_tokens is not None:
        extra_body["min_tokens"] = args.min_tokens
    records = await client.run_concurrent(
        [args.prompt] * args.concurrency,
        args.max_tokens,
        args.concurrency,
        args.stream,
        endpoint=args.endpoint,
        extra_body=extra_body,
    )
    print(json.dumps(records_to_rows(records), indent=2))


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
