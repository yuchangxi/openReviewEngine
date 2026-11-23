from __future__ import annotations

import json

import pytest

from deep_review.config import ToolLimits
from deep_review.llm import ToolCall
from deep_review.tools.repo_tools import ToolContext, default_tools, execute, tools_by_name

TOOLS = tools_by_name(default_tools())


def run(ctx: ToolContext, name: str, **args) -> tuple[str, bool]:
    return execute(TOOLS[name], ctx, ToolCall(id="t", name=name, arguments=json.dumps(args)))


@pytest.fixture
def ctx(ws) -> ToolContext:
    return ToolContext.for_workspace(ws)


def test_read_file_window_and_paging(ws) -> None:
    (ws.root / "long.py").write_text("".join(f"x{i} = {i}\n" for i in range(1, 301)))
    ctx = ToolContext.for_workspace(ws, ToolLimits(read_window_lines=50))
    out, ok = run(ctx, "read_file", path="long.py", start_line=10)
    assert ok
    assert out.splitlines()[0] == "long.py (lines 10-59 of 300)"
    assert "    10| x10 = 10" in out and "    59| x59 = 59" in out and "x60" not in out
    assert "start_line=60" in out
    out, ok = run(ctx, "read_file", path="long.py", start_line=400)
    assert not ok and "only 300 lines" in out


def test_tool_errors_are_returned_not_raised(ctx) -> None:
    out, ok = run(ctx, "read_file", path="../../etc/passwd")
    assert not ok and out.startswith("ERROR:") and "outside the repository" in out
    out, ok = run(ctx, "read_file", path="nope.py")
    assert not ok and "no such file" in out
    out, ok = run(ctx, "read_file", path="img.bin")
    assert not ok and "binary" in out
    out, ok = run(ctx, "read_file")  # missing required arg
    assert not ok and "path: Field required" in out
    out, ok = execute(TOOLS["grep"], ctx, ToolCall(id="t", name="grep", arguments='{"pattern":'))
    assert not ok and "not valid JSON" in out
    out, ok = run(ctx, "read_file", path="new.py", bogus=1)
    assert not ok and "bogus" in out


def test_grep_and_find_refs(ctx) -> None:
    out, ok = run(ctx, "grep", pattern=r"def \w+\(")
    assert ok and "2 matches in 1 files" in out and "lib/util.py:1: def add(a, b, c=0):" in out
    out, ok = run(ctx, "grep", pattern="add", glob="new.py")
    assert ok and "lib/util.py" not in out and "new.py:1:" in out
    out, ok = run(ctx, "grep", pattern="(unclosed")
    assert not ok and "search failed" in out
    out, ok = run(ctx, "grep", pattern="zzz_nothing")
    assert ok and out == "no matches found"
    out, ok = run(ctx, "grep", pattern="x", path="../")
    assert not ok and "outside" in out

    out, ok = run(ctx, "find_refs", symbol="add")
    assert ok and "new.py:1:" in out and "new.py:3:" in out and "lib/util.py:1:" in out
    out, ok = run(ctx, "find_refs", symbol="util.add")  # qualified name → last part
    assert ok and "new.py:3:" in out
    out, ok = run(ctx, "find_refs", symbol="a b")
    assert not ok and "use grep" in out


def test_grep_caps_results(ws) -> None:
    (ws.root / "many.txt").write_text("hit\n" * 100)
    ctx = ToolContext.for_workspace(ws, ToolLimits(max_grep_results=5))
    out, ok = run(ctx, "grep", pattern="hit")
    assert ok and "100 matches in 1 files (showing first 5" in out
    assert len(out.splitlines()) == 6


def test_list_dir(ctx) -> None:
    out, ok = run(ctx, "list_dir")
    assert ok and out.splitlines() == ["./", "lib/", "a.py", "img.bin", "new.py"]
    out, ok = run(ctx, "list_dir", path="new.py")
    assert not ok and "not a directory" in out


def test_git_show_and_blame(ctx, upstream) -> None:
    out, ok = run(ctx, "git_show", rev="base", path="lib/util.py", end_line=2)
    assert ok and "lib/util.py @ base (lines 1-2 of 6)" in out and "def add(a, b):" in out
    out, ok = run(ctx, "git_show", rev="head", path="lib/util.py", end_line=1)
    assert ok and "def add(a, b, c=0):" in out
    out, ok = run(ctx, "git_show", rev="base", path="new.py")
    assert not ok and "does not exist at base" in out
    out, ok = run(ctx, "git_show", rev="HEAD~1", path="a.py")
    assert not ok and "unknown rev" in out

    out, ok = run(ctx, "git_blame", path="lib/util.py", start_line=1, end_line=5)
    assert ok
    lines = out.splitlines()[1:]
    assert len(lines) == 5
    assert lines[0].startswith(upstream["feature_commit"][:7])  # changed by the PR
    assert lines[4].lstrip("^").startswith(upstream["base_commit"][:7])  # untouched (^ = root)
    out, ok = run(ctx, "git_blame", path="lib/util.py", start_line=1, end_line=500)
    assert not ok and "at most 100" in out


def test_output_is_capped(ws) -> None:
    (ws.root / "wide.py").write_text("".join(f"{'y' * 300}{i}\n" for i in range(100)))
    ctx = ToolContext.for_workspace(ws, ToolLimits(max_tool_output_chars=2_000))
    out, ok = run(ctx, "read_file", path="wide.py")
    assert ok and len(out) < 2_100 and out.endswith("request a smaller window]")


def test_specs_are_compact_and_valid() -> None:
    for tool in default_tools():
        params = tool.spec.parameters
        assert params["type"] == "object" and "title" not in json.dumps(params)
        assert params.get("additionalProperties") is False
