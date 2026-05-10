from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from agentweaver.utils.hashing import prompt_hash
from agentweaver.utils.io import ensure_dir
from agentweaver.utils.tokenization import count_tokens


def _render_prompt(body: dict[str, Any]) -> str:
    if isinstance(body.get("messages"), list):
        return "\n".join(str(m.get("content", "")) for m in body["messages"] if isinstance(m, dict))
    return str(body.get("prompt") or body.get("input") or "")


def _extract_text_from_response(obj: dict[str, Any]) -> str:
    texts: list[str] = []
    for choice in obj.get("choices", []) or []:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                texts.append(content)
        text = choice.get("text")
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts)


def _usage(obj: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = obj.get("usage") if isinstance(obj, dict) else None
    if not isinstance(usage, dict):
        return None, None
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
    completion = usage.get("completion_tokens") or usage.get("output_tokens")
    return (int(prompt) if prompt is not None else None, int(completion) if completion is not None else None)


def _sse_delta_text(data: str) -> str:
    data = data.strip()
    if not data or data == "[DONE]":
        return ""
    try:
        obj = json.loads(data)
    except Exception:
        return ""
    chunks: list[str] = []
    for choice in obj.get("choices", []) or []:
        delta = choice.get("delta") if isinstance(choice, dict) else None
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            chunks.append(delta["content"])
        if isinstance(choice, dict) and isinstance(choice.get("text"), str):
            chunks.append(choice["text"])
    return "".join(chunks)


class TimingLogger:
    def __init__(self, path: str | Path, tokenizer: str):
        self.path = Path(path)
        self.tokenizer = tokenizer
        ensure_dir(self.path.parent)
        self._lock = asyncio.Lock()

    async def write(self, row: dict[str, Any]) -> None:
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")


def create_app(backend: str, log_path: str, tokenizer: str) -> FastAPI:
    app = FastAPI()
    backend = backend.rstrip("/")
    logger = TimingLogger(log_path, tokenizer)
    client = httpx.AsyncClient(timeout=None)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await client.aclose()

    @app.get("/v1/models")
    async def models() -> Response:
        resp = await client.get(f"{backend}/models")
        return Response(content=resp.content, status_code=resp.status_code, headers={"content-type": resp.headers.get("content-type", "application/json")})

    @app.get("/metrics")
    async def metrics() -> Response:
        metrics_url = backend.rsplit("/v1", 1)[0] + "/metrics" if backend.endswith("/v1") else backend + "/metrics"
        resp = await client.get(metrics_url)
        return Response(content=resp.content, status_code=resp.status_code, headers={"content-type": resp.headers.get("content-type", "text/plain")})

    async def forward_json(request: Request, path: str) -> Response:
        body = await request.json()
        request_id = str(uuid.uuid4())
        rendered = _render_prompt(body)
        h = prompt_hash(rendered)
        model = str(body.get("model") or "")
        start = time.time()
        resp = await client.post(f"{backend}{path}", json=body, headers={k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}})
        end = time.time()
        content_type = resp.headers.get("content-type", "application/json")
        out_text = ""
        prompt_tokens = None
        completion_tokens = None
        try:
            obj = resp.json()
            out_text = _extract_text_from_response(obj)
            prompt_tokens, completion_tokens = _usage(obj)
        except Exception:
            obj = {}
        input_tokens = prompt_tokens if prompt_tokens is not None else count_tokens(rendered, tokenizer)
        output_tokens = completion_tokens if completion_tokens is not None else count_tokens(out_text, tokenizer)
        await logger.write(
            {
                "request_id": request_id,
                "endpoint": path,
                "prompt_hash": h,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "timestamp_start": start,
                "first_token_time": None,
                "timestamp_end": end,
                "latency": max(0.0, end - start),
                "ttft": None,
                "tpot": None,
                "model": model,
                "streaming": False,
                "status_code": resp.status_code,
            }
        )
        return Response(content=resp.content, status_code=resp.status_code, headers={"content-type": content_type})

    async def forward_stream(request: Request, path: str) -> StreamingResponse:
        body = await request.json()
        request_id = str(uuid.uuid4())
        rendered = _render_prompt(body)
        h = prompt_hash(rendered)
        model = str(body.get("model") or "")
        start = time.time()
        first_token_time: float | None = None
        chunks_text: list[str] = []
        status_code = 200

        async def gen():
            nonlocal first_token_time, status_code
            async with client.stream("POST", f"{backend}{path}", json=body, headers={k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}}) as resp:
                status_code = resp.status_code
                async for chunk in resp.aiter_bytes():
                    now = time.time()
                    if first_token_time is None and chunk:
                        first_token_time = now
                    text = chunk.decode("utf-8", errors="replace")
                    for line in text.splitlines():
                        if line.startswith("data:"):
                            delta = _sse_delta_text(line[5:])
                            if delta:
                                chunks_text.append(delta)
                    yield chunk
            end = time.time()
            output_text = "".join(chunks_text)
            output_tokens = count_tokens(output_text, tokenizer)
            ttft = None if first_token_time is None else max(0.0, first_token_time - start)
            decode = None if first_token_time is None else max(0.0, end - first_token_time)
            await logger.write(
                {
                    "request_id": request_id,
                    "endpoint": path,
                    "prompt_hash": h,
                    "input_tokens": count_tokens(rendered, tokenizer),
                    "output_tokens": output_tokens,
                    "timestamp_start": start,
                    "first_token_time": first_token_time,
                    "timestamp_end": end,
                    "latency": max(0.0, end - start),
                    "ttft": ttft,
                    "tpot": (decode / output_tokens) if decode is not None and output_tokens else None,
                    "model": model,
                    "streaming": True,
                    "status_code": status_code,
                }
            )

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        async def replay_body() -> Request:
            return request
        request._json = body  # type: ignore[attr-defined]
        if body.get("stream"):
            return await forward_stream(request, "/chat/completions")
        return await forward_json(request, "/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        request._json = body  # type: ignore[attr-defined]
        if body.get("stream"):
            return await forward_stream(request, "/completions")
        return await forward_json(request, "/completions")

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--backend", default="http://localhost:8001/v1")
    ap.add_argument("--log-path", required=True)
    ap.add_argument("--tokenizer-path", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    args = ap.parse_args()
    uvicorn.run(create_app(args.backend, args.log_path, args.tokenizer_path), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
