"""Render diffs for the model with explicit new-side line numbers.

Weaker models miscount lines in raw unified diffs, so every added/context line carries its
head line number — the number findings must cite. Deleted lines carry none.

    @@ -10,3 +10,4 @@ def handle():
       10   a
       11 + b
          - old
       12   c
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from deep_review.diff.parser import FileDiff
from deep_review.llm.tokens import estimate_tokens

_TAG = {"add": "+", "del": "-", "ctx": " "}


def render_file(fd: FileDiff) -> str:
    if fd.status == "renamed":
        header = f"### {fd.new_path} (renamed from {fd.old_path})"
    else:
        header = f"### {fd.path} ({fd.status})"
    out = [header]
    for h in fd.hunks:
        out.append(
            f"@@ -{h.old_start},{h.old_len} +{h.new_start},{h.new_len} @@ {h.section}".rstrip()
        )
        for ln in h.lines:
            num = f"{ln.new_no:>6}" if ln.new_no is not None else " " * 6
            out.append(f"{num} {_TAG[ln.kind]} {ln.text}")
    return "\n".join(out)


@dataclass
class RenderedDiff:
    text: str
    shown: list[str] = field(default_factory=list)
    omitted: list[str] = field(default_factory=list)
    tokens: int = 0


def render_diff(files: Sequence[FileDiff], max_tokens: int) -> RenderedDiff:
    """Render files in order until the token budget is reached; list the rest as omitted.
    Smaller files go first so one huge file cannot crowd out everything else."""
    out = RenderedDiff(text="")
    parts: list[str] = []
    for fd in sorted(files, key=lambda f: f.additions + f.deletions):
        block = render_file(fd)
        cost = estimate_tokens(block)
        if out.tokens + cost > max_tokens:
            out.omitted.append(fd.path)
            continue
        parts.append(block)
        out.shown.append(fd.path)
        out.tokens += cost
    out.text = "\n\n".join(parts)
    return out
