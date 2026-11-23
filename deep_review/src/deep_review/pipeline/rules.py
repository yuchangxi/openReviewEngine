"""Repository rules discovery (PLAN §2 item 2): the root rules files plus rules files in the
directories of modified files (and their parents)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import PurePosixPath

from deep_review.llm.tokens import estimate_tokens
from deep_review.tools.pathguard import PathGuard

RULE_FILES = (
    "AGENTS.md",
    "CLAUDE.md",
    "REVIEW.md",
    "CODE_REVIEW.md",
    ".cursorrules",
    ".github/copilot-instructions.md",
)


def discover_rules(guard: PathGuard, changed: Iterable[str], max_tokens: int) -> str:
    dirs: list[PurePosixPath] = [PurePosixPath(".")]
    for path in changed:
        for parent in reversed(PurePosixPath(path).parents):
            if parent not in dirs:
                dirs.append(parent)
    parts: list[str] = []
    used = 0
    for d in dirs:
        for name in RULE_FILES:
            rel = (d / name).as_posix()
            try:
                text = guard.read_text(rel, 200_000).strip()
            except Exception:
                continue
            if not text:
                continue
            cost = estimate_tokens(text)
            if used + cost > max_tokens:
                text = text[: int((max_tokens - used) * 3)] + "\n…[truncated]"
                cost = max_tokens - used
            parts.append(f"## {rel}\n{text}")
            used += cost
            if used >= max_tokens:
                return "\n\n".join(parts)
    return "\n\n".join(parts)
