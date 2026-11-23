"""Bounded agent loop (PLAN §4.3): `run_agent`.

Guarantees, so a weak or misbehaving model cannot run away:
- turn cap and context-token budget; old tool outputs are elided before the budget is hit;
- identical repeated tool calls are refused after `max_repeats`;
- tool arguments are validated, and errors go back to the model as tool results;
- the agent ends by calling the submit tool; its arguments are validated against a schema;
- when a limit is reached, one final turn offers *only* the submit tool, so partial results
  are kept. (`tool_choice` forcing is unavailable in DeepSeek thinking mode, so this is prompted.)
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ValidationError

from deep_review.errors import LLMError, ToolArgumentError
from deep_review.llm.tokens import estimate_messages, estimate_tokens
from deep_review.llm.types import LLM, Message, ToolCall, ToolSpec, Usage
from deep_review.tools.repo_tools import Tool, ToolContext, execute, tool_spec, tools_by_name
from deep_review.trace import Tracer

StopReason = Literal["submitted", "max_turns", "token_budget", "no_submission", "llm_error"]

ELIDED = "[older tool output removed to save context; call the tool again if you still need it]"
KEEP_RECENT_TOOL_OUTPUTS = 6
KEEP_MIN_TOOL_OUTPUTS = 2
ELIDE_AT = 0.75  # fraction of the context budget


@dataclass(frozen=True)
class SubmitTool:
    name: str
    description: str
    model: type[BaseModel]

    @property
    def spec(self) -> ToolSpec:
        return tool_spec(self.name, self.description, self.model)


@dataclass
class AgentResult:
    submission: BaseModel | None
    stop_reason: StopReason
    forced_finish: bool = False
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    usage: Usage = field(default_factory=Usage)
    peak_context_tokens: int = 0
    messages: list[Message] = field(default_factory=list)
    error: str | None = None


def _call_key(call: ToolCall) -> tuple[str, str]:
    try:
        args = json.dumps(json.loads(call.arguments or "{}"), sort_keys=True)
    except json.JSONDecodeError:
        args = call.arguments
    return call.name, args


def _add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        cached_prompt_tokens=a.cached_prompt_tokens + b.cached_prompt_tokens,
    )


def _elide_old_tool_outputs(messages: list[Message], target_tokens: int) -> int:
    """Replace old tool outputs with a marker. First keep only the most recent
    KEEP_RECENT_TOOL_OUTPUTS; if the estimate is still above target, keep going down to
    KEEP_MIN_TOOL_OUTPUTS (a few huge outputs can matter more than many small ones)."""

    def live() -> list[int]:
        return [i for i, m in enumerate(messages) if m.role == "tool" and m.content != ELIDED]

    def elide(i: int) -> None:
        messages[i] = Message.tool(messages[i].tool_call_id or "", ELIDED)

    n = 0
    idx = live()
    for i in idx[: max(len(idx) - KEEP_RECENT_TOOL_OUTPUTS, 0)]:
        elide(i)
        n += 1
    while (
        estimate_messages(messages) > target_tokens and len(idx := live()) > KEEP_MIN_TOOL_OUTPUTS
    ):
        elide(idx[0])
        n += 1
    return n


def _validate_submission(submit: SubmitTool, call: ToolCall) -> BaseModel:
    raw = call.parse_arguments()
    try:
        return submit.model.model_validate(raw)
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:10]
        )
        raise ToolArgumentError(f"submission rejected: {problems}. Fix and call again.") from e


def run_agent(
    llm: LLM,
    tools: Sequence[Tool],
    ctx: ToolContext,
    messages: list[Message],
    submit: SubmitTool,
    *,
    max_turns: int,
    context_budget_tokens: int,
    max_output_tokens: int | None = None,
    context_window_tokens: int | None = None,
    final_max_output_tokens: int = 32_000,
    warn_turns_left: int = 3,
    max_repeats: int = 2,
    tracer: Tracer | None = None,
) -> AgentResult:
    """`context_window_tokens` is the model's hard limit (prompt + output). With it, the
    output allowance grows when a reply was cut off (`finish_reason=length`, which reasoning
    models hit by thinking past the cap) and for the final submit turn, but never past the
    space actually left in the window."""
    tr: Tracer = tracer or Tracer.null()
    by_name = tools_by_name(tools)
    specs = [t.spec for t in tools] + [submit.spec]
    result = AgentResult(submission=None, stop_reason="no_submission", messages=messages)
    seen: Counter[tuple[str, str]] = Counter()
    nudged = False
    context_tokens = estimate_messages(messages)
    base_out = max_output_tokens or 8_000
    boost_next = False

    def allowance(want: int) -> int:
        if context_window_tokens is None:
            return want
        room = context_window_tokens - estimate_messages(messages) - 2_000
        return max(min(want, room), 1_000)

    def chat(offered: list[ToolSpec], out_tokens: int | None = None) -> Message | None:
        nonlocal context_tokens, boost_next
        t0 = time.perf_counter()
        want = out_tokens or (final_max_output_tokens if boost_next else base_out)
        try:
            resp = llm.chat(messages, tools=offered, max_tokens=allowance(want))
        except LLMError as e:
            result.stop_reason, result.error = "llm_error", str(e)
            tr.event("llm_error", error=str(e))
            return None
        result.turns += 1
        result.usage = _add_usage(result.usage, resp.usage)
        context_tokens = resp.usage.prompt_tokens or estimate_messages(messages)
        context_tokens += resp.usage.completion_tokens or estimate_messages([resp.message])
        result.peak_context_tokens = max(result.peak_context_tokens, context_tokens)
        tr.event(
            "llm_turn",
            turn=result.turns,
            finish_reason=resp.finish_reason,
            usage=resp.usage.model_dump(),
            latency_s=round(time.perf_counter() - t0, 2),
            content=resp.message.content,
            tool_calls=[tc.model_dump() for tc in resp.message.tool_calls],
            reasoning_chars=len(resp.message.reasoning or ""),
        )
        boost_next = resp.finish_reason == "length" and not resp.message.tool_calls
        if resp.message.content or resp.message.tool_calls:
            # An empty assistant message (cut off while reasoning) is not kept: some
            # endpoints reject a message with neither content nor tool calls.
            messages.append(resp.message.model_copy(update={"reasoning": None}))
        return resp.message

    def try_submit(call: ToolCall) -> bool:
        try:
            result.submission = _validate_submission(submit, call)
        except ToolArgumentError as e:
            result.tool_errors += 1
            messages.append(Message.tool(call.id, f"ERROR: {e}"))
            tr.event("submit_rejected", error=str(e))
            return False
        result.stop_reason = "submitted"
        tr.event("submitted", arguments_chars=len(call.arguments))
        return True

    while result.turns < max_turns:
        msg = chat(specs)
        if msg is None:
            return result
        if not msg.tool_calls:
            if nudged:  # second plain-text reply: give up on this model's turn
                break
            nudged = True
            messages.append(
                Message.user(
                    "Do not answer in plain text. Continue investigating with the tools, or "
                    f"finish by calling {submit.name} (use an empty list if you found nothing)."
                )
            )
            continue
        for call in msg.tool_calls:
            if call.name == submit.name:
                if try_submit(call):
                    return result
                continue
            result.tool_calls += 1
            tool = by_name.get(call.name)
            key = _call_key(call)
            seen[key] += 1
            t0 = time.perf_counter()
            if tool is None:
                out, ok = (
                    f"ERROR: unknown tool {call.name!r}; available: "
                    + ", ".join([*by_name, submit.name]),
                    False,
                )
            elif seen[key] > max_repeats:
                out, ok = (
                    f"ERROR: you already made this exact call {seen[key] - 1} times and its "
                    "result has not changed. Try something different or submit.",
                    False,
                )
            else:
                out, ok = execute(tool, ctx, call)
            result.tool_errors += not ok
            messages.append(Message.tool(call.id, out))
            context_tokens += estimate_tokens(out)
            tr.event(
                "tool_call",
                tool=call.name,
                arguments=call.arguments,
                ok=ok,
                output_chars=len(out),
                output_preview=out[:300],
                latency_s=round(time.perf_counter() - t0, 3),
            )
        if context_tokens > ELIDE_AT * context_budget_tokens:
            n = _elide_old_tool_outputs(messages, int(ELIDE_AT * context_budget_tokens))
            if n:
                context_tokens = estimate_messages(messages)
                tr.event("elided_tool_outputs", count=n, context_tokens=context_tokens)
        if context_tokens > context_budget_tokens:
            result.stop_reason = "token_budget"
            break
        left = max_turns - result.turns
        if left == warn_turns_left:
            messages.append(
                Message.user(
                    f"You have {left} turns left. Wrap up the most important checks and call "
                    f"{submit.name} soon; an unsubmitted review is lost."
                )
            )
    else:
        result.stop_reason = "max_turns"

    # Graceful stop: offer only the submit tool, with a generous output allowance, and retry
    # once if the reply is empty or cut off — so the investigation is not thrown away.
    if result.stop_reason in ("max_turns", "token_budget", "no_submission"):
        reason = result.stop_reason
        result.forced_finish = True
        prompts = [
            f"Stop investigating now: your {'turn' if reason == 'max_turns' else 'context'} "
            f"budget is used up. Call {submit.name} now with the findings you have, each with "
            "an honest confidence. Keep your reasoning short.",
            f"Your last reply was empty or cut off. Call {submit.name} immediately with the "
            "findings you already have. Do not deliberate further.",
        ]
        for text in prompts:
            messages.append(Message.user(text))
            msg = chat([submit.spec], out_tokens=final_max_output_tokens)
            if msg is None:
                return result
            for c in msg.tool_calls:
                if c.name == submit.name:
                    if try_submit(c):
                        return result
                else:
                    # Every tool call needs a tool response, or the next request is rejected.
                    messages.append(
                        Message.tool(c.id, f"ERROR: only {submit.name} is available now; call it.")
                    )
            tr.event("final_turn_without_submission", finish_reason_empty=not msg.content)
        result.stop_reason = reason
    return result
