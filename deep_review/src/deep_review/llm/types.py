"""Provider-neutral chat types. The rest of the engine depends only on these and `LLM`."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from deep_review.errors import StructuredOutputError, ToolArgumentError

Role = Literal["system", "user", "assistant", "tool"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolCall(_Frozen):
    id: str
    name: str
    arguments: str = "{}"
    """Raw JSON string exactly as the model produced it."""

    def parse_arguments(self) -> dict[str, Any]:
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError as e:
            raise ToolArgumentError(
                f"arguments for tool {self.name!r} are not valid JSON ({e.msg} at pos {e.pos})"
            ) from e
        if not isinstance(value, dict):
            raise ToolArgumentError(f"arguments for tool {self.name!r} must be a JSON object")
        return value


class Message(_Frozen):
    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    reasoning: str | None = None
    """Provider reasoning text (e.g. `reasoning_content`). Kept for traces; not sent back."""

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | None = None, tool_calls: Sequence[ToolCall] = ()) -> Message:
        return cls(role="assistant", content=content, tool_calls=tuple(tool_calls))

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Message:
        return cls(role="tool", tool_call_id=tool_call_id, content=content)

    def to_openai(self) -> dict[str, Any]:
        if self.role == "assistant":
            d: dict[str, Any] = {"role": "assistant", "content": self.content}
            if self.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in self.tool_calls
                ]
            return d
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.content or "",
            }
        return {"role": self.role, "content": self.content or ""}


class Usage(_Frozen):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatResponse(_Frozen):
    message: Message
    finish_reason: str = "stop"
    usage: Usage = Field(default_factory=Usage)
    model: str = ""


class ToolSpec(_Frozen):
    name: str
    description: str
    parameters: dict[str, Any]
    """JSON schema of the arguments object."""

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class LLM(Protocol):
    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse: ...


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def parse_structured[T: BaseModel](response: ChatResponse, model: type[T]) -> T:
    """Validate the response content against `model`. Tolerates a ```json fence."""
    text = response.message.content or ""
    if m := _FENCE_RE.match(text):
        text = m[1]
    if not text.strip():
        raise StructuredOutputError(
            f"empty response where {model.__name__} JSON was expected "
            f"(finish_reason={response.finish_reason})"
        )
    try:
        return model.model_validate_json(text)
    except ValidationError as e:
        raise StructuredOutputError(f"response does not match {model.__name__}: {e}") from e
