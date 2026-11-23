"""Thin OpenAI-compatible chat client (internal vLLM gateway, or a dev endpoint for public code)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx2
import openai
from pydantic import BaseModel

from deep_review.config import LLMConfig
from deep_review.errors import ConfigError, LLMError
from deep_review.llm.types import ChatResponse, Message, ToolCall, ToolSpec, Usage


class OpenAICompatClient:
    def __init__(
        self,
        cfg: LLMConfig,
        *,
        model: str | None = None,
        api_key: str | None = None,
        http_client: httpx2.Client | None = None,
    ) -> None:
        if not cfg.base_url:
            raise ConfigError("llm.base_url is not set (or export DEEP_REVIEW_BASE_URL)")
        self.model = model or cfg.finder_model
        if not self.model:
            raise ConfigError("no model configured (llm.finder_model or DEEP_REVIEW_MODEL)")
        self.cfg = cfg
        self._client = openai.OpenAI(
            base_url=cfg.base_url,
            api_key=api_key or cfg.api_key(),
            timeout=cfg.timeout_s,
            max_retries=cfg.max_retries,
            http_client=http_client,
        )

    def _request(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        response_model: type[BaseModel] | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload = [m.to_openai() for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "temperature": self.cfg.temperature,
            "max_tokens": max_tokens or self.cfg.max_output_tokens,
        }
        if tools:
            kwargs["tools"] = [t.to_openai() for t in tools]
        if response_model is not None:
            schema = response_model.model_json_schema()
            if self.cfg.structured_output == "json_schema":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": response_model.__name__, "schema": schema},
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}
                # Appended last so the shared prompt prefix stays cacheable.
                payload.append(
                    {
                        "role": "user",
                        "content": "Reply with one JSON object that matches this JSON schema:\n"
                        + json.dumps(schema, ensure_ascii=False),
                    }
                )
        return kwargs

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        kwargs = self._request(messages, tools, response_model, max_tokens)
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except openai.APIError as e:
            raise LLMError(f"chat completion failed ({type(e).__name__}): {e}") from e
        if not resp.choices:
            raise LLMError("chat completion returned no choices")
        choice = resp.choices[0]
        msg = choice.message
        tool_calls = tuple(
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (msg.tool_calls or [])
            if getattr(tc, "type", "function") == "function"
        )
        return ChatResponse(
            message=Message(
                role="assistant",
                content=msg.content,
                tool_calls=tool_calls,
                reasoning=getattr(msg, "reasoning_content", None),
            ),
            finish_reason=choice.finish_reason or "stop",
            usage=_usage(resp.usage),
            model=resp.model or self.model,
        )


def _usage(u: Any) -> Usage:
    if u is None:
        return Usage()
    details = getattr(u, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details else None
    if cached is None:  # DeepSeek's own API reports cache hits this way
        cached = getattr(u, "prompt_cache_hit_tokens", None)
    return Usage(
        prompt_tokens=u.prompt_tokens or 0,
        completion_tokens=u.completion_tokens or 0,
        cached_prompt_tokens=cached or 0,
    )
