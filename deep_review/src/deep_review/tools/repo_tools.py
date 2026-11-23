"""Read-only repository tools for agents (PLAN §4.3), designed for weaker models (SWE-agent ACI):
windowed file views with line numbers, capped search results, and errors that say what to do next.

No tool runs a shell or PR code. Paths go through PathGuard; git revisions are limited to the
pinned `base` (merge-base) and `head` SHAs.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from deep_review.config import ToolLimits
from deep_review.errors import GitError, ToolArgumentError, ToolError
from deep_review.llm.types import ToolCall, ToolSpec
from deep_review.tools.pathguard import PathGuard, is_lfs_pointer, looks_binary
from deep_review.workspace.git import run_git

_MAX_LINE_CHARS = 400
_IDENT_RE = re.compile(r"^[A-Za-z_$][\w$.:]*$")


@dataclass
class ToolContext:
    guard: PathGuard
    git_dir: Path | None
    revs: dict[str, str]
    """Allowed revision aliases, e.g. {"base": <merge-base sha>, "head": <head sha>}."""
    limits: ToolLimits = field(default_factory=ToolLimits)
    rg_path: str | None = field(default_factory=lambda: shutil.which("rg"))
    timeout_s: float = 30.0

    @classmethod
    def for_workspace(cls, ws: Any, limits: ToolLimits | None = None) -> ToolContext:
        return cls(
            guard=ws.guard,
            git_dir=ws.git_dir,
            revs={"base": ws.merge_base, "head": ws.head_sha},
            limits=limits or ToolLimits(),
        )


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- helpers


def _clip(line: str) -> str:
    return line if len(line) <= _MAX_LINE_CHARS else line[:_MAX_LINE_CHARS] + " …[line truncated]"


def _window(text: str, label: str, start: int, end: int | None, max_lines: int) -> str:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    total = len(lines)
    if total == 0:
        return f"{label}: (empty file)"
    if start > total:
        raise ToolError(f"{label} has only {total} lines; start_line {start} is past the end")
    end = total if end is None else min(end, total)
    if end < start:
        raise ToolError(f"end_line ({end}) is before start_line ({start})")
    end = min(end, start + max_lines - 1)
    body = "\n".join(f"{n:>6}| {_clip(lines[n - 1])}" for n in range(start, end + 1))
    more = (
        f"\n[showing lines {start}-{end} of {total}; call again with start_line={end + 1} for more]"
        if end < total
        else ""
    )
    return f"{label} (lines {start}-{end} of {total})\n{body}{more}"


def _resolve_rev(ctx: ToolContext, rev: str) -> str:
    if rev not in ctx.revs:
        raise ToolError(f"unknown rev {rev!r}; use one of: {', '.join(sorted(ctx.revs))}")
    return ctx.revs[rev]


def _repo_rel(ctx: ToolContext, path: str) -> str:
    return ctx.guard.relpath(ctx.guard.resolve(path)) if path not in ("", ".") else "."


def _run_rg(ctx: ToolContext, args: list[str]) -> list[dict[str, Any]]:
    if ctx.rg_path is None:
        raise ToolError("search is unavailable (ripgrep not installed on the review host)")
    cmd = [ctx.rg_path, "--json", "--no-config", "--line-number", "--glob", "!.git", *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=ctx.guard.root,
            capture_output=True,
            timeout=ctx.timeout_s,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as e:
        raise ToolError("search timed out; narrow the pattern or the path") from e
    if proc.returncode not in (0, 1):  # 1 = no matches
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise ToolError(f"search failed: {err[-500:]}")
    matches = []
    for line in proc.stdout.splitlines():
        event = json.loads(line)
        if event.get("type") == "match":
            data = event["data"]
            text = data["lines"].get("text")
            if text is None:  # non-UTF-8 line
                continue
            matches.append(
                {
                    "path": data["path"]["text"].removeprefix("./"),
                    "line": data["line_number"],
                    "text": text.rstrip("\n"),
                }
            )
    return matches


def _format_matches(matches: list[dict[str, Any]], cap: int, what: str) -> str:
    if not matches:
        return f"no {what} found"
    shown = matches[:cap]
    body = "\n".join(f"{m['path']}:{m['line']}: {_clip(m['text'].strip())}" for m in shown)
    files = len({m["path"] for m in matches})
    head = f"{len(matches)} {what} in {files} files"
    if len(matches) > cap:
        head += f" (showing first {cap}; narrow with path/glob)"
    return f"{head}\n{body}"


# ---------------------------------------------------------------- tools


class ReadFileArgs(_Args):
    path: str = Field(description="File path relative to the repository root")
    start_line: int = Field(1, ge=1, description="First line to show (1-based)")
    end_line: int | None = Field(None, ge=1, description="Last line to show (inclusive)")


def read_file(ctx: ToolContext, a: ReadFileArgs) -> str:
    text = ctx.guard.read_text(a.path, ctx.limits.max_file_bytes)
    return _window(
        text, _repo_rel(ctx, a.path), a.start_line, a.end_line, ctx.limits.read_window_lines
    )


class GrepArgs(_Args):
    pattern: str = Field(min_length=1, description="Regular expression (Rust regex syntax)")
    path: str = Field(".", description="Directory or file to search, relative to the repo root")
    glob: str | None = Field(None, description="Only search files matching this glob, e.g. '*.py'")
    fixed_string: bool = Field(False, description="Treat pattern as a literal string")
    ignore_case: bool = False


def grep(ctx: ToolContext, a: GrepArgs) -> str:
    target = _repo_rel(ctx, a.path)
    args = ["--max-columns", "1000"]
    if a.fixed_string:
        args.append("--fixed-strings")
    if a.ignore_case:
        args.append("--ignore-case")
    if a.glob:
        if a.glob.startswith("!") and ".git" in a.glob:
            raise ToolArgumentError("invalid glob")
        args += ["--glob", a.glob]
    args += ["--regexp", a.pattern, "--", target]
    return _format_matches(_run_rg(ctx, args), ctx.limits.max_grep_results, "matches")


class ListDirArgs(_Args):
    path: str = Field(".", description="Directory relative to the repository root")


def list_dir(ctx: ToolContext, a: ListDirArgs) -> str:
    d = ctx.guard.resolve(a.path)
    if not d.is_dir():
        raise ToolError(f"not a directory: {a.path}")
    entries = sorted(
        (p for p in d.iterdir() if p.name != ".git"), key=lambda p: (not p.is_dir(), p.name)
    )
    cap = 300
    lines = [p.name + ("/" if p.is_dir() else "") for p in entries[:cap]]
    if len(entries) > cap:
        lines.append(f"… {len(entries) - cap} more entries")
    return f"{_repo_rel(ctx, a.path)}/\n" + ("\n".join(lines) if lines else "(empty)")


class FindRefsArgs(_Args):
    symbol: str = Field(description="Identifier to look up, e.g. 'parse_token' or 'Signer.sign'")
    glob: str | None = Field(None, description="Only search files matching this glob")


def find_refs(ctx: ToolContext, a: FindRefsArgs) -> str:
    if not _IDENT_RE.match(a.symbol):
        raise ToolArgumentError(f"symbol must be an identifier, got {a.symbol!r}; use grep instead")
    name = re.split(r"[.:]+", a.symbol)[-1]  # Signer.sign → sign
    args = ["--fixed-strings", "--word-regexp", "--regexp", name]
    if a.glob:
        args += ["--glob", a.glob]
    matches = _run_rg(ctx, [*args, "--", "."])
    out = _format_matches(matches, ctx.limits.max_grep_results, f"textual references to {name!r}")
    return out + "\n[text search: may include comments, strings and unrelated same-name symbols]"


class GitShowArgs(_Args):
    rev: str = Field(description="'base' (the PR's merge-base) or 'head' (the PR head)")
    path: str = Field(description="File path relative to the repository root")
    start_line: int = Field(1, ge=1)
    end_line: int | None = Field(None, ge=1)


def git_show(ctx: ToolContext, a: GitShowArgs) -> str:
    sha = _resolve_rev(ctx, a.rev)
    rel = _repo_rel(ctx, a.path)
    if ctx.git_dir is None:
        raise ToolError("git history is unavailable")
    proc = run_git(
        ["cat-file", "blob", f"{sha}:{rel}"],
        git_dir=ctx.git_dir,
        timeout=ctx.timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        raise ToolError(f"{rel} does not exist at {a.rev}")
    data = proc.stdout
    if len(data) > ctx.limits.max_file_bytes:
        raise ToolError(f"{rel} at {a.rev} is too large ({len(data)} bytes)")
    if looks_binary(data) or is_lfs_pointer(data):
        raise ToolError(f"{rel} at {a.rev} is binary")
    text = data.decode("utf-8", errors="replace")
    return _window(text, f"{rel} @ {a.rev}", a.start_line, a.end_line, ctx.limits.read_window_lines)


class GitBlameArgs(_Args):
    path: str = Field(description="File path relative to the repository root")
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


def git_blame(ctx: ToolContext, a: GitBlameArgs) -> str:
    if a.end_line < a.start_line:
        raise ToolArgumentError("end_line is before start_line")
    if a.end_line - a.start_line >= 100:
        raise ToolArgumentError("blame at most 100 lines at a time")
    if ctx.git_dir is None:
        raise ToolError("git history is unavailable")
    rel = _repo_rel(ctx, a.path)
    try:
        out = run_git(
            ["blame", "-s", "-L", f"{a.start_line},{a.end_line}", ctx.revs["head"], "--", rel],
            git_dir=ctx.git_dir,
            timeout=ctx.timeout_s * 4,
        ).stdout.decode("utf-8", errors="replace")
    except GitError as e:
        raise ToolError(f"blame failed for {rel}:{a.start_line}-{a.end_line}: {e}") from e
    return f"blame {rel} @ head (commit, line, text)\n" + "\n".join(
        _clip(ln) for ln in out.splitlines()
    )


# ---------------------------------------------------------------- registry


def _clean_schema(schema: Any) -> Any:
    """Drop pydantic 'title' noise: fewer tokens, same meaning."""
    if isinstance(schema, dict):
        return {k: _clean_schema(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_clean_schema(v) for v in schema]
    return schema


def tool_spec(name: str, description: str, args_model: type[BaseModel]) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        parameters=_clean_schema(args_model.model_json_schema()),
    )


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    args_model: type[BaseModel]
    handler: Callable[[ToolContext, Any], str]

    @property
    def name(self) -> str:
        return self.spec.name


def _tool(name: str, description: str, model: type[BaseModel], fn: Callable) -> Tool:
    return Tool(tool_spec(name, description, model), model, fn)


def default_tools() -> list[Tool]:
    return [
        _tool(
            "read_file",
            "Show a window of a file at the PR head, with line numbers. "
            "Up to 200 lines per call; page with start_line.",
            ReadFileArgs,
            read_file,
        ),
        _tool(
            "grep",
            "Regex search across the repository at the PR head. Returns path:line: text, "
            "capped; narrow with path or glob.",
            GrepArgs,
            grep,
        ),
        _tool("list_dir", "List a directory at the PR head.", ListDirArgs, list_dir),
        _tool(
            "find_refs",
            "Find textual references to an identifier (whole-word match) at the "
            "PR head, e.g. callers of a changed function.",
            FindRefsArgs,
            find_refs,
        ),
        _tool(
            "git_show",
            "Show a window of a file at 'base' (before the PR) or 'head'. Use it to "
            "compare old and new behaviour.",
            GitShowArgs,
            git_show,
        ),
        _tool(
            "git_blame",
            "Blame up to 100 lines at the PR head, to see which lines are new.",
            GitBlameArgs,
            git_blame,
        ),
    ]


def parse_args(tool: Tool, call: ToolCall) -> BaseModel:
    raw = call.parse_arguments()
    try:
        return tool.args_model.model_validate(raw)
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(map(str, err['loc'])) or 'arguments'}: {err['msg']}" for err in e.errors()
        )
        raise ToolArgumentError(f"invalid arguments for {tool.name}: {problems}") from e


def execute(tool: Tool, ctx: ToolContext, call: ToolCall) -> tuple[str, bool]:
    """Run one tool call. Returns (output for the model, ok). Never raises ToolError."""
    try:
        out = tool.handler(ctx, parse_args(tool, call))
        ok = True
    except ToolError as e:
        out, ok = f"ERROR: {e}", False
    cap = ctx.limits.max_tool_output_chars
    if len(out) > cap:
        out = out[:cap] + f"\n[output truncated at {cap} characters; request a smaller window]"
    return out, ok


def tools_by_name(tools: Sequence[Tool]) -> dict[str, Tool]:
    return {t.name: t for t in tools}
