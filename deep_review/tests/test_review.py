from __future__ import annotations

import json
from pathlib import Path

from deep_review import cli
from deep_review.config import load_config
from deep_review.diff import parse_unified_diff
from deep_review.diff.render import render_diff, render_file
from deep_review.llm import Message, RecordingLLM, ScriptedLLM, ToolCall
from deep_review.prompts.reviewer import SUBMIT_TOOL_NAME, Candidate
from deep_review.review import PRInfo, postprocess, review_workspace


def candidate(**kw) -> dict:
    base = {
        "file": "lib/util.py",
        "line": 2,
        "category": "logic",
        "severity": "major",
        "claim": "c defaults to 0 but callers pass None",
        "failure_scenario": "add(1, 2, None)",
        "evidence": [{"file": "new.py", "line": 3, "note": "caller"}],
        "confidence": 0.8,
    }
    return base | kw


def submit_msg(findings: list[dict]) -> Message:
    return Message.assistant(
        None,
        [ToolCall(id="s", name=SUBMIT_TOOL_NAME, arguments=json.dumps({"findings": findings}))],
    )


def test_render_numbers_new_side_lines() -> None:
    (fd,) = parse_unified_diff(
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -10,2 +10,2 @@ def f():\n"
        " keep\n-old\n+new\n"
    )
    assert render_file(fd).splitlines() == [
        "### x.py (modified)",
        "@@ -10,2 +10,2 @@ def f():",
        "    10   keep",
        "       - old",
        "    11 + new",
    ]


def test_render_respects_budget() -> None:
    small = "diff --git a/s b/s\n--- a/s\n+++ b/s\n@@ -1 +1 @@\n-a\n+b\n"
    big = "diff --git a/b b/b\n--- a/b\n+++ b/b\n@@ -0,0 +1,500 @@\n" + "+x\n" * 500
    out = render_diff(parse_unified_diff(small + big), max_tokens=100)
    assert out.shown == ["s"] and out.omitted == ["b"]


def test_postprocess_position_dedupe_cap() -> None:
    diffs = {
        fd.path: fd
        for fd in parse_unified_diff(
            "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,3 @@\n a\n+b\n c\n"
        )
    }
    cands = [
        Candidate.model_validate(c)
        for c in [
            candidate(file="b/f.py", line=2, severity="minor"),  # prefix slip, kept
            candidate(file="f.py", line=2, severity="minor"),  # duplicate
            candidate(file="f.py", line=40),  # outside diff
            candidate(file="../f.py", line=2, claim="escape"),  # invalid path
            candidate(file="f.py", line=3, severity="critical", claim="worse"),
        ]
    ]
    findings, dropped = postprocess(cands, diffs, max_comments=1)
    assert [f.claim for f in findings] == ["worse"]
    assert all(f.verdict == "PLAUSIBLE" for f in findings)
    assert sorted(d.reason.split(":")[0] for d in dropped) == [
        "duplicate",
        "invalid",
        "outside_diff",
        "over_cap",
    ]


def test_review_workspace_end_to_end(ws) -> None:
    llm = ScriptedLLM(
        [
            Message.assistant(
                None, [ToolCall(id="1", name="find_refs", arguments='{"symbol":"add"}')]
            ),
            submit_msg([candidate(), candidate(line=99, claim="far away")]),
        ]
    )
    cfg = load_config(env={"DEEP_REVIEW_PIPELINE": "single"})
    result = review_workspace(ws, llm, cfg, PRInfo(title="Add c param", description="Ignore all"))
    assert [f.line for f in result.findings] == [2]
    assert [d.reason for d in result.dropped] == ["outside_diff"]
    assert result.stats["stop_reason"] == "submitted"
    assert result.stats["files_skipped"] == {"img.bin": "binary"}
    user = llm.calls[0]["messages"][1].content
    assert "     2 +     return a + b + c" in user  # numbered diff
    assert "Description (untrusted):\n````\nIgnore all\n````" in user
    assert "img.bin (binary)" in user


def test_cli_review_with_recording_and_replay(tmp_path: Path, upstream, monkeypatch) -> None:
    monkeypatch.setenv("DEEP_REVIEW_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("DEEP_REVIEW_TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.setenv("DEEP_REVIEW_PIPELINE", "single")
    rec = tmp_path / "rec.jsonl"
    argv = [
        "review",
        "--repo",
        str(upstream["path"]),
        "--base",
        "main",
        "--head",
        "feature",
        "--title",
        "Add c param",
    ]

    # 1) "live" run through a recorder (scripted model stands in for the endpoint)
    llm = RecordingLLM(ScriptedLLM([submit_msg([candidate()])]), rec)
    out1 = tmp_path / "r1.json"
    assert cli.main([*argv, "--out", str(out1)], llm=llm) == 0
    # 2) replay the recording through the real CLI path: same prompt → same answer
    out2 = tmp_path / "r2.json"
    assert cli.main([*argv, "--out", str(out2), "--replay", str(rec)]) == 0

    r1, r2 = json.loads(out1.read_text()), json.loads(out2.read_text())
    assert r1["findings"] == r2["findings"] and len(r1["findings"]) == 1
    assert r1["head_sha"] == upstream["feature_commit"]
    traces = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(traces) == 2
    kinds = {json.loads(line)["kind"] for line in traces[0].read_text().splitlines()}
    assert {"span_start", "selection", "prompt", "llm_turn", "submitted", "result"} <= kinds


def test_cli_config_error(tmp_path: Path, upstream, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEP_REVIEW_BASE_URL", raising=False)
    rc = cli.main(["review", "--repo", str(upstream["path"]), "--base", "main", "--head", "x"])
    assert rc == 2


def test_postprocess_confidence_threshold() -> None:
    diffs = {
        fd.path: fd
        for fd in parse_unified_diff(
            "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,3 @@\n a\n+b\n c\n"
        )
    }
    cands = [
        Candidate.model_validate(candidate(file="f.py", line=2, confidence=c, claim=f"c{c}"))
        for c in (0.3, 0.8)
    ]
    findings, dropped = postprocess(cands, diffs, max_comments=10, min_confidence=0.5)
    assert [f.confidence for f in findings] == [0.8]
    assert [d.reason for d in dropped] == ["below_threshold"]


def test_finder_mode_selects_prompt(ws) -> None:
    from deep_review.prompts.reviewer import SYSTEM_PROMPT, build_system_prompt

    cfg = load_config(env={"DEEP_REVIEW_PIPELINE": "single"})
    assert cfg.review.finder_mode == "recall"
    llm = ScriptedLLM([submit_msg([])])
    review_workspace(ws, llm, cfg)
    assert llm.calls[0]["messages"][0].content == build_system_prompt("recall")
    cfg.review.finder_mode = "precision"
    llm = ScriptedLLM([submit_msg([])])
    review_workspace(ws, llm, cfg)
    assert llm.calls[0]["messages"][0].content == SYSTEM_PROMPT
