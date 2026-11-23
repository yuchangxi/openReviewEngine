"""Cheap token estimates for budgeting when the endpoint does not report usage (e.g. mocks).

Code tokenizes denser than prose; ~3 characters per token is a deliberately conservative guess.
Real usage from the API always takes precedence.
"""

from __future__ import annotations

from collections.abc import Iterable

from deep_review.llm.types import Message

CHARS_PER_TOKEN = 3.0


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def estimate_messages(messages: Iterable[Message]) -> int:
    total = 0
    for m in messages:
        total += 4 + estimate_tokens(m.content or "")
        for tc in m.tool_calls:
            total += 4 + estimate_tokens(tc.name + tc.arguments)
    return total
