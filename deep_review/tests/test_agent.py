from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from deep_review.agent import SubmitTool, run_agent
from deep_review.agent.loop import ELIDED
from deep_review.errors import LLMError
from deep_review.llm import ChatResponse, Message, ScriptedLLM, ToolCall, Usage
from deep_review.tools.repo_tools import ToolContext, default_tools


class Answer(BaseModel):
    items: list[str]


SUBMIT = SubmitTool("submit", "Submit the answer", Answer)


def call(name: str, i: int = 0, **args) -> Message:
    return Message.assistant(None, [ToolCall(id=f"c{i}", name=name, arguments=json.dumps(args))])


def submit(items: list[str], i: int = 99) -> Message:
    return call("submit", i, items=items)


@pytest.fixture
def ctx(ws) -> ToolContext:
    return ToolContext.for_workspace(ws)


def go(llm, ctx, *, max_turns: int = 10, budget: int = 50_000, **kw):
    msgs = [Message.system("sys"), Message.user("task")]
    return run_agent(
        llm,
        default_tools(),
        ctx,
        msgs,
        SUBMIT,
        max_turns=max_turns,
        context_budget_tokens=budget,
        **kw,
    )


def test_happy_path(ctx) -> None:
    llm = ScriptedLLM([call("read_file", 1, path="new.py"), submit(["found"])])
    r = go(llm, ctx)
    assert r.stop_reason == "submitted" and r.submission == Answer(items=["found"])
    assert r.turns == 2 and r.tool_calls == 1 and r.tool_errors == 0 and not r.forced_finish
    tool_msg = r.messages[3]
    assert (
        tool_msg.role == "tool"
        and tool_msg.tool_call_id == "c1"
        and "from lib.util" in tool_msg.content
    )
    # every request offered the repo tools plus submit
    assert [t.name for t in llm.calls[0]["tools"]][-1] == "submit"


def test_repeated_calls_are_refused(ctx) -> None:
    same = [call("read_file", i, path="new.py") for i in range(4)]
    r = go(ScriptedLLM([*same, submit([])]), ctx)
    outs = [m.content for m in r.messages if m.role == "tool"]
    assert "from lib.util" in outs[0] and "from lib.util" in outs[1]
    assert outs[2].startswith("ERROR: you already made this exact call 2 times")
    assert r.tool_errors == 2 and r.stop_reason == "submitted"


def test_plain_text_gets_one_nudge(ctx) -> None:
    r = go(ScriptedLLM(["I think it is fine.", submit([])]), ctx)
    assert r.stop_reason == "submitted" and r.turns == 2
    assert "Do not answer in plain text" in r.messages[3].content


def test_two_plain_replies_force_a_submit_only_turn(ctx) -> None:
    llm = ScriptedLLM(["fine", "still fine", submit(["late"])])
    r = go(llm, ctx)
    assert r.forced_finish and r.stop_reason == "submitted" and r.submission.items == ["late"]
    assert [t.name for t in llm.calls[-1]["tools"]] == ["submit"]


def test_bad_submission_is_returned_for_correction(ctx) -> None:
    bad = call("submit", 1, items="not-a-list")
    r = go(ScriptedLLM([bad, submit(["ok"])]), ctx)
    assert r.stop_reason == "submitted" and r.tool_errors == 1
    assert "submission rejected" in r.messages[3].content


def test_unknown_tool(ctx) -> None:
    r = go(ScriptedLLM([call("rm_rf", 1), submit([])]), ctx)
    assert "unknown tool 'rm_rf'" in r.messages[3].content


def test_turn_cap_keeps_partial_results(ctx) -> None:
    loop = [call("grep", i, pattern=f"p{i}") for i in range(3)]
    llm = ScriptedLLM([*loop, submit(["partial"])])
    r = go(llm, ctx, max_turns=3)
    assert r.forced_finish and r.stop_reason == "submitted" and r.submission.items == ["partial"]
    assert r.turns == 4  # 3 + the submit-only turn
    assert "turn budget is used up" in r.messages[-2].content


def test_turn_cap_without_submission(ctx) -> None:
    llm = ScriptedLLM([call("grep", 0, pattern="a"), call("grep", 1, pattern="b"), "no", "no"])
    r = go(llm, ctx, max_turns=2)
    assert r.forced_finish and r.submission is None and r.stop_reason == "max_turns"
    assert r.turns == 4  # two final attempts, then give up


def _turn(msg: Message, prompt_tokens: int) -> ChatResponse:
    return ChatResponse(
        message=msg, usage=Usage(prompt_tokens=prompt_tokens), finish_reason="tool_calls"
    )


def test_token_budget_forces_a_submit_only_turn(ctx) -> None:
    llm = ScriptedLLM(
        [
            _turn(call("grep", 1, pattern="a"), 10_000),
            _turn(call("grep", 2, pattern="b"), 60_000),  # over the 50k budget
            submit(["budget"]),
        ]
    )
    r = go(llm, ctx, budget=50_000)
    assert r.stop_reason == "submitted" and r.forced_finish and r.turns == 3
    assert "context budget is used up" in r.messages[-2].content
    assert r.peak_context_tokens >= 60_000
    assert [t.name for t in llm.calls[-1]["tools"]] == ["submit"]


def test_old_tool_outputs_are_elided_near_the_budget(ctx) -> None:
    reads = [_turn(call("grep", i, pattern=f"p{i}"), 5_000) for i in range(8)]
    llm = ScriptedLLM([*reads, _turn(call("grep", 8, pattern="last"), 40_000), submit([])])
    r = go(llm, ctx, budget=50_000)  # 40k > 75% of 50k → elide, but no forced stop
    assert r.stop_reason == "submitted" and not r.forced_finish
    tool_msgs = [m for m in llm.calls[-1]["messages"] if m.role == "tool"]
    assert [m.content == ELIDED for m in tool_msgs] == [True] * 3 + [False] * 6


def test_huge_outputs_are_elided_down_to_the_minimum(ctx, ws) -> None:
    (ws.root / "huge.txt").write_text(("z" * 100 + "\n") * 110)
    # identical calls would be refused, so vary the path spelling
    # prompt_tokens=0 → the loop estimates context size from the messages themselves
    reads = [_turn(call("read_file", i, path="./" * i + "huge.txt"), 0) for i in range(4)]
    llm = ScriptedLLM([*reads, submit([])])
    r = go(llm, ctx, budget=10_000)  # each output ≈ 3.7k tokens by estimate
    assert r.stop_reason == "submitted"
    tool_msgs = [m for m in llm.calls[-1]["messages"] if m.role == "tool"]
    assert sum(m.content != ELIDED for m in tool_msgs) == 2


def test_llm_error_is_reported(ctx) -> None:
    class Broken:
        def chat(self, *a, **k):
            raise LLMError("gateway down")

    r = go(Broken(), ctx)
    assert r.stop_reason == "llm_error" and "gateway down" in r.error and r.submission is None


def test_reasoning_is_not_kept_in_history(ctx) -> None:
    m = call("read_file", 1, path="new.py").model_copy(update={"reasoning": "long thoughts"})
    llm = ScriptedLLM([m, submit([])])
    r = go(llm, ctx)
    assert all(msg.reasoning is None for msg in r.messages)


def _cut_off() -> ChatResponse:
    """A reasoning model that thought past its output cap: no content, no tool calls."""
    return ChatResponse(
        message=Message.assistant(None),
        finish_reason="length",
        usage=Usage(prompt_tokens=1_000, completion_tokens=8_000),
    )


def test_final_turn_retries_after_cut_off_reply(ctx) -> None:
    loop = [call("grep", i, pattern=f"p{i}") for i in range(2)]
    llm = ScriptedLLM([*loop, _cut_off(), submit(["rescued"])])
    r = go(llm, ctx, max_turns=2, max_output_tokens=8_000, final_max_output_tokens=32_000)
    assert r.stop_reason == "submitted" and r.submission.items == ["rescued"]
    assert [c["max_tokens"] for c in llm.calls] == [8_000, 8_000, 32_000, 32_000]
    # the empty assistant message was not put into the history sent back to the model
    assert not any(
        m.role == "assistant" and not m.content and not m.tool_calls
        for m in llm.calls[-1]["messages"]
    )
    assert "empty or cut off" in llm.calls[-1]["messages"][-1].content


def test_output_allowance_never_exceeds_the_window(ctx) -> None:
    llm = ScriptedLLM([submit([])])
    go(llm, ctx, max_output_tokens=8_000, context_window_tokens=5_000)
    assert llm.calls[0]["max_tokens"] <= 5_000 - 2_000 and llm.calls[0]["max_tokens"] >= 1_000


def test_cut_off_turn_boosts_the_next_allowance(ctx) -> None:
    llm = ScriptedLLM([_cut_off(), submit([])])
    r = go(llm, ctx, max_output_tokens=8_000, final_max_output_tokens=24_000)
    assert r.stop_reason == "submitted"
    assert [c["max_tokens"] for c in llm.calls] == [8_000, 24_000]


def test_turns_left_warning(ctx) -> None:
    loop = [call("grep", i, pattern=f"p{i}") for i in range(3)]
    llm = ScriptedLLM([*loop, submit([])])
    go(llm, ctx, max_turns=5, warn_turns_left=3)
    warned = [
        m
        for m in llm.calls[2]["messages"]
        if m.role == "user" and "turns left" in (m.content or "")
    ]
    assert len(warned) == 1 and "3 turns left" in warned[0].content


def test_final_turn_answers_disallowed_tool_calls(ctx) -> None:
    """Regression: a non-submit call in the final turn left an unanswered tool_call → HTTP 400."""
    loop = [call("grep", i, pattern=f"p{i}") for i in range(2)]
    llm = ScriptedLLM([*loop, call("read_file", 50, path="new.py"), submit(["ok"])])
    r = go(llm, ctx, max_turns=2)
    assert r.stop_reason == "submitted" and r.submission.items == ["ok"]
    history = llm.calls[-1]["messages"]
    for i, m in enumerate(history):  # every assistant tool call is answered before the next turn
        for tc in m.tool_calls:
            assert any(h.role == "tool" and h.tool_call_id == tc.id for h in history[i + 1 :])
