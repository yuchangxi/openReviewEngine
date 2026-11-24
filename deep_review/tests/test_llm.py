from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx2
import pytest
from pydantic import BaseModel

from deep_review.config import LLMConfig
from deep_review.errors import LLMError, StructuredOutputError, ToolArgumentError
from deep_review.llm import (
    ChatResponse,
    Message,
    OpenAICompatClient,
    RecordingLLM,
    ReplayLLM,
    ScriptedLLM,
    ToolCall,
    ToolSpec,
    parse_structured,
)
from deep_review.llm.mock import MockExhaustedError, ReplayMismatchError

READ_FILE = ToolSpec(
    name="read_file",
    description="Read a file window",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
)


class Verdict(BaseModel):
    verdict: str
    confidence: float


def _completion(message: dict, *, usage: dict | None = None, finish: str = "stop") -> dict:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "served-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _client(handler, **cfg) -> tuple[OpenAICompatClient, list[dict]]:
    seen: list[dict] = []

    def wrapped(request: httpx2.Request) -> httpx2.Response:
        seen.append(
            {
                "url": str(request.url),
                "auth": request.headers.get("authorization"),
                "headers": dict(request.headers),
                "body": json.loads(request.content),
            }
        )
        return handler(request)

    config = LLMConfig(
        base_url="http://gateway.local/v1", finder_model="flash", max_retries=0, **cfg
    )
    client = OpenAICompatClient(
        config,
        api_key="test-key",
        http_client=httpx2.Client(transport=httpx2.MockTransport(wrapped)),
    )
    return client, seen


def test_client_tool_call_round_trip() -> None:
    reply = _completion(
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "thinking...",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                }
            ],
        },
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 7,
            "total_tokens": 107,
            "prompt_cache_hit_tokens": 64,
        },
        finish="tool_calls",
    )
    client, seen = _client(lambda r: httpx2.Response(200, json=reply))
    history = [
        Message.system("sys"),
        Message.user("review"),
        Message.assistant(None, [ToolCall(id="c0", name="read_file", arguments='{"path":"b.py"}')]),
        Message.tool("c0", "contents of b"),
    ]
    resp = client.chat(history, tools=[READ_FILE], max_tokens=256)

    body = seen[0]["body"]
    assert seen[0]["url"] == "http://gateway.local/v1/chat/completions"
    assert seen[0]["auth"] == "Bearer test-key"
    assert body["model"] == "flash" and body["max_tokens"] == 256 and body["temperature"] == 0.0
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert body["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"path":"b.py"}'
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "c0", "content": "contents of b"}

    assert resp.finish_reason == "tool_calls"
    assert resp.message.tool_calls[0].parse_arguments() == {"path": "a.py"}
    assert resp.message.reasoning == "thinking..."
    assert resp.usage.cached_prompt_tokens == 64 and resp.usage.total_tokens == 107


@pytest.mark.parametrize("mode", ["json_schema", "json_object"])
def test_client_structured_output_modes(mode: str) -> None:
    reply = _completion(
        {"role": "assistant", "content": '{"verdict": "CONFIRMED", "confidence": 0.9}'}
    )
    client, seen = _client(lambda r: httpx2.Response(200, json=reply), structured_output=mode)
    resp = client.chat([Message.user("verify")], response_model=Verdict)
    body = seen[0]["body"]
    assert body["response_format"]["type"] == mode
    if mode == "json_object":
        assert "JSON schema" in body["messages"][-1]["content"]
    else:
        assert body["response_format"]["json_schema"]["schema"]["title"] == "Verdict"
    assert parse_structured(resp, Verdict) == Verdict(verdict="CONFIRMED", confidence=0.9)


def test_client_identifies_itself_and_sends_stable_session() -> None:
    ok = lambda r: httpx2.Response(200, json=_completion({"role": "assistant", "content": "x"}))  # noqa: E731
    client, seen = _client(ok, session_header="x-opencode-session")
    shared = [Message.system("sys"), Message.user("pr diff")]
    client.chat([*shared, Message.user("focus: logic")])
    client.chat([*shared, Message.user("focus: security"), Message.user("more")])
    client.chat([Message.system("sys"), Message.user("other pr")])
    ids = [s["headers"]["x-opencode-session"] for s in seen]
    assert ids[0] == ids[1] != ids[2]  # same conversation prefix → same session
    assert all(s["headers"]["user-agent"] == "deep-review/0.1" for s in seen)

    plain, seen2 = _client(ok)
    plain.chat(shared)
    assert "x-opencode-session" not in seen2[0]["headers"]


def test_client_wraps_http_errors() -> None:
    client, _ = _client(lambda r: httpx2.Response(503, json={"error": {"message": "overloaded"}}))
    with pytest.raises(LLMError, match="overloaded"):
        client.chat([Message.user("x")])


def test_parse_structured_and_tool_args() -> None:
    fenced = ChatResponse(message=Message.assistant('```json\n{"verdict":"X","confidence":1}\n```'))
    assert parse_structured(fenced, Verdict).verdict == "X"
    with pytest.raises(StructuredOutputError, match="empty"):
        parse_structured(ChatResponse(message=Message.assistant("")), Verdict)
    with pytest.raises(StructuredOutputError):
        parse_structured(ChatResponse(message=Message.assistant('{"verdict": 1}')), Verdict)
    with pytest.raises(ToolArgumentError, match="not valid JSON"):
        ToolCall(id="1", name="grep", arguments='{"pattern": ').parse_arguments()
    with pytest.raises(ToolArgumentError, match="JSON object"):
        ToolCall(id="1", name="grep", arguments="[1]").parse_arguments()


def test_scripted_llm() -> None:
    llm = ScriptedLLM(["first", Message.assistant(None, [ToolCall(id="1", name="grep")])])
    assert llm.chat([Message.user("a")]).message.content == "first"
    assert llm.chat([Message.user("b")]).finish_reason == "tool_calls"
    assert len(llm.calls) == 2
    with pytest.raises(MockExhaustedError):
        llm.chat([Message.user("c")])


def test_record_then_replay_in_any_order(tmp_path: Path) -> None:
    rec_path = tmp_path / "rec.jsonl"
    inner = ScriptedLLM(["answer-a", "answer-b"])
    rec = RecordingLLM(inner, rec_path)
    rec.chat([Message.user("a")], tools=[READ_FILE])
    rec.chat([Message.user("b")])

    replay = ReplayLLM(rec_path)
    results: dict[str, str | None] = {}

    def ask(q: str, tools: list[ToolSpec]) -> None:
        results[q] = replay.chat([Message.user(q)], tools=tools).message.content

    threads = [
        threading.Thread(target=ask, args=("b", [])),
        threading.Thread(target=ask, args=("a", [READ_FILE])),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == {"a": "answer-a", "b": "answer-b"}

    with pytest.raises(ReplayMismatchError):
        replay.chat([Message.user("a")])  # same prompt but tools differ → different request
