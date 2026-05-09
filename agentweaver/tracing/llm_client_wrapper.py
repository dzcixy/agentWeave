from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import requests

from agentweaver.tracing.prompt_segmenter import segment_prompt
from agentweaver.tracing.trace_schema import ContextSegmentRef, Event, Trace
from agentweaver.utils.hashing import prompt_hash, stable_hash
from agentweaver.utils.tokenization import count_tokens


class OpenAICompatibleTracer:
    def __init__(self, base_url: str, api_key: str = "EMPTY", model: str = "qwen-coder-7b", trace_path: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.trace_path = trace_path

    def chat(self, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> dict[str, Any]:
        url = self.base_url + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        rendered = "\n".join(str(m.get("content", "")) for m in messages)
        segs = segment_prompt(messages, metadata={"model": self.model})
        t0 = time.time()
        resp = requests.post(url, headers=headers, json={"model": self.model, "messages": messages, "stream": stream, **kwargs}, timeout=600)
        t1 = time.time()
        resp.raise_for_status()
        data = resp.json() if not stream else {"stream": True}
        usage = data.get("usage", {})
        out_text = ""
        try:
            out_text = data["choices"][0]["message"]["content"]
        except Exception:
            pass
        ev = Event(
            event_id=f"llm-{stable_hash([rendered, t0])}",
            session_id=str(kwargs.get("session_id", "llm-session")),
            instance_id=str(kwargs.get("instance_id", "unknown")),
            branch_id=str(kwargs.get("branch_id", "b0")),
            parent_branch_id=kwargs.get("parent_branch_id"),
            step_id=int(kwargs.get("step_id", 0)),
            node_id=f"llm-{stable_hash([rendered, t0], 8)}",
            node_type="llm",
            role=str(kwargs.get("role", "worker")),
            model=self.model,
            timestamp_ready=t0,
            timestamp_start=t0,
            timestamp_end=t1,
            latency=t1 - t0,
            input_tokens=int(usage.get("prompt_tokens", count_tokens(rendered))),
            output_tokens=int(usage.get("completion_tokens", count_tokens(out_text))),
            context_length=int(usage.get("prompt_tokens", count_tokens(rendered))),
            prompt_hash=prompt_hash(rendered),
            shared_prefix_id=segs[0].segment_id if segs else None,
            context_segments=[ContextSegmentRef(s.segment_id, s.segment_type, s.start_pos, s.length) for s in segs],
            context_segment_defs=segs,
            ttft=None,
        )
        if self.trace_path:
            Trace({"framework": "llm_client_wrapper", "timestamp": t0, "model": self.model}, [ev]).to_jsonl(self.trace_path)
        return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="qwen-coder-7b")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out-trace")
    args = ap.parse_args()
    c = OpenAICompatibleTracer(args.base_url, args.api_key, args.model, args.out_trace)
    print(json.dumps(c.chat([{"role": "user", "content": args.prompt}], max_tokens=128), indent=2))


if __name__ == "__main__":
    main()
