"""Mock LLMs for development without the internal model (PLAN §3).

- `ScriptedLLM`: returns canned responses in order; for unit tests.
- `RecordingLLM`: wraps a real client and appends every exchange to a JSONL file.
- `ReplayLLM`: serves a recording back, matched by request fingerprint, so replay is
  deterministic even when specialists run in parallel.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from deep_review.errors import LLMError
from deep_review.llm.types import LLM, ChatResponse, Message, ToolSpec


class MockExhaustedError(LLMError):
    pass


class ReplayMismatchError(LLMError):
    pass


def request_fingerprint(
    messages: Sequence[Message],
    tools: Sequence[ToolSpec] = (),
    response_model: type[BaseModel] | None = None,
) -> str:
    """Hash of everything that determines the model's answer, except the model name."""
    canonical = {
        "messages": [m.to_openai() for m in messages],
        "tools": [t.to_openai() for t in tools],
        "response_schema": response_model.model_json_schema() if response_model else None,
    }
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _as_response(item: ChatResponse | Message | str) -> ChatResponse:
    if isinstance(item, ChatResponse):
        return item
    if isinstance(item, Message):
        return ChatResponse(message=item, finish_reason="tool_calls" if item.tool_calls else "stop")
    return ChatResponse(message=Message.assistant(item))


class ScriptedLLM:
    def __init__(self, responses: Iterable[ChatResponse | Message | str]) -> None:
        self._queue: deque[ChatResponse] = deque(_as_response(r) for r in responses)
        self._lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        with self._lock:
            self.calls.append(
                {
                    "messages": list(messages),
                    "tools": list(tools),
                    "response_model": response_model,
                    "max_tokens": max_tokens,
                }
            )
            if not self._queue:
                raise MockExhaustedError(
                    f"ScriptedLLM ran out of responses at call {len(self.calls)}"
                )
            return self._queue.popleft()


class RecordingLLM:
    def __init__(self, inner: LLM, path: Path | str) -> None:
        self.inner = inner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        resp = self.inner.chat(
            messages, tools=tools, response_model=response_model, max_tokens=max_tokens
        )
        record = {
            "fingerprint": request_fingerprint(messages, tools, response_model),
            "request": {
                "messages": [m.to_openai() for m in messages],
                "tools": [t.name for t in tools],
                "response_model": response_model.__name__ if response_model else None,
            },
            "response": resp.model_dump(mode="json"),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        return resp


class ReplayLLM:
    def __init__(self, path: Path | str) -> None:
        self._by_fp: dict[str, deque[ChatResponse]] = defaultdict(deque)
        self._lock = threading.Lock()
        for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                self._by_fp[rec["fingerprint"]].append(ChatResponse.model_validate(rec["response"]))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                raise LLMError(f"{path}:{n}: bad recording line: {e}") from e

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        fp = request_fingerprint(messages, tools, response_model)
        with self._lock:
            queue = self._by_fp.get(fp)
            if not queue:
                raise ReplayMismatchError(
                    f"no recorded response for request {fp[:12]} — prompts, tools or schema "
                    "changed since recording; re-record against a real endpoint"
                )
            return queue.popleft()
