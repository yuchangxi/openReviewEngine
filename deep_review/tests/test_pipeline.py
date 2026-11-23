from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable

from deep_review.config import load_config
from deep_review.diff import parse_unified_diff
from deep_review.findings import Finding
from deep_review.llm import ChatResponse, Message, ToolCall
from deep_review.pipeline import run_pipeline
from deep_review.pipeline.dedupe import claim_similarity, dedupe
from deep_review.pipeline.graph import _chunk_files
from deep_review.pipeline.rules import discover_rules
from deep_review.prompts.reviewer import SUBMIT_TOOL_NAME
from deep_review.prompts.verifier import SUBMIT_VERDICT, VERIFIER_SYSTEM
from deep_review.review import PRInfo, review_workspace


class RoutingLLM:
    """Answers by role: specialists are recognised by their focus message, verifiers by the
    system prompt. Thread-safe, records every request."""

    def __init__(self, route: Callable[[list[Message]], Message]) -> None:
        self.route = route
        self.calls: list[list[Message]] = []
        self._lock = threading.Lock()

    def chat(self, messages, *, tools=(), response_model=None, max_tokens=None) -> ChatResponse:
        with self._lock:
            self.calls.append(list(messages))
        msg = self.route(list(messages))
        return ChatResponse(message=msg, finish_reason="tool_calls" if msg.tool_calls else "stop")


def submit(findings: list[dict]) -> Message:
    return Message.assistant(
        None,
        [ToolCall(id="s", name=SUBMIT_TOOL_NAME, arguments=json.dumps({"findings": findings}))],
    )


def verdict(v: str, conf: float, reason: str = "checked") -> Message:
    args = {
        "verdict": v,
        "confidence": conf,
        "reason": reason,
        "evidence": [{"file": "new.py", "line": 3, "note": "caller"}],
    }
    return Message.assistant(
        None, [ToolCall(id="v", name=SUBMIT_VERDICT, arguments=json.dumps(args))]
    )


def cand(file: str, line: int, claim: str, category: str = "logic", conf: float = 0.7) -> dict:
    return {
        "file": file,
        "line": line,
        "category": category,
        "severity": "major",
        "claim": claim,
        "failure_scenario": "add(1, 2) with old callers",
        "confidence": conf,
    }


SIG = "add() gained a third parameter c that changes the sum for existing callers"


def role(messages: list[Message]) -> str:
    if messages[0].content == VERIFIER_SYSTEM:
        return "verifier"
    focus = messages[2].content or ""
    for name, marker in [
        ("logic", "logic and correctness"),
        ("contract", "contracts, callers"),
        ("concurrency_resource", "concurrency, transactions"),
        ("security", "Your focus: security"),
        ("rules", "rules compliance"),
    ]:
        if marker in focus:
            return name
    raise AssertionError(f"unknown role: {focus[:80]}")


def standard_route(messages: list[Message]) -> Message:
    r = role(messages)
    if r == "logic":
        return submit([cand("lib/util.py", 2, SIG, conf=0.6)])
    if r == "contract":
        return submit([cand("lib/util.py", 1, SIG + " silently", category="contract", conf=0.8)])
    if r == "concurrency_resource":
        return submit([cand("a.py", 1, "outside the diff")])
    if r == "security":
        return submit([cand("new.py", 1, "imports from an untrusted module", "security")])
    if r == "rules":
        return submit([])
    report = messages[1].content or ""
    return (
        verdict("CONFIRMED", 0.9) if "lib/util.py" in report else verdict("REJECTED", 0.1, "fine")
    )


def cfg(**pipeline):
    c = load_config(env={})
    for k, v in pipeline.items():
        setattr(c.pipeline, k, v)
    return c


def test_full_pipeline(ws) -> None:
    llm = RoutingLLM(standard_route)
    r = run_pipeline(ws, llm, cfg(), PRInfo(title="Add c"))

    # logic + contract reported the same issue → merged; security → rejected; a.py → outside diff
    assert [(f.file, f.line, f.verdict, f.confidence) for f in r.findings] == [
        ("lib/util.py", 1, "CONFIRMED", 0.9)
    ]
    assert {f.file for f in r.pre_verification} == {"lib/util.py", "new.py"}
    assert next(f for f in r.pre_verification if f.file == "lib/util.py").confidence == 0.8
    reasons = sorted(d.reason.split(":")[0] for d in r.dropped)
    assert reasons == ["merged_into", "outside_diff", "rejected"]
    assert any(e.file == "new.py" and e.line == 3 for e in r.findings[0].evidence)  # verifier's

    s = r.stats
    assert s["pipeline"] == "graph" and s["candidates"] == 4 and s["verified"] == 2
    assert s["verdict_counts"] == {"CONFIRMED": 1, "REJECTED": 1}
    assert s["specialists"] == ["logic", "contract", "concurrency_resource", "security"]
    assert s["agent_stop_reasons"] == {"specialist:submitted": 4, "verifier:submitted": 2}

    spec_calls = [m for m in llm.calls if m[0].content != VERIFIER_SYSTEM]
    assert len(spec_calls) == 4
    # identical shared prefix (system + PR/diff) for every specialist → prefix-cache friendly
    assert len({(m[0].content, m[1].content) for m in spec_calls}) == 1
    # verifiers start fresh: no specialist focus or reasoning in their context
    for m in (m for m in llm.calls if m[0].content == VERIFIER_SYSTEM):
        assert len(m) == 2 and "Your focus" not in (m[1].content or "")


def test_verification_off_and_missing_verdicts(ws) -> None:
    c = cfg(verify=False)
    c.aggregate.plausible_threshold = 0.0
    r = run_pipeline(ws, RoutingLLM(standard_route), c)
    assert {f.verdict for f in r.findings} == {"PLAUSIBLE"} and len(r.findings) == 2
    assert {f.id for f in r.findings} == {f.id for f in r.pre_verification}
    assert r.stats["verdict_counts"] == {}

    def silent_verifier(messages):
        if role(messages) == "verifier":
            return Message.assistant("I am not sure.")  # never submits
        return standard_route(messages)

    c = cfg(verifier_max_turns=2)
    c.aggregate.plausible_threshold = 0.0
    r = run_pipeline(ws, RoutingLLM(silent_verifier), c)
    assert r.stats["verdict_counts"] == {"no_verdict": 2}
    assert len(r.findings) == 2 and all(f.confidence <= 0.5 for f in r.findings)  # capped


def test_threshold_keeps_confirmed_and_drops_weak_plausible(ws) -> None:
    def route(messages):
        if role(messages) == "verifier":
            report = messages[1].content or ""
            return (
                verdict("CONFIRMED", 0.6) if "lib/util.py" in report else verdict("PLAUSIBLE", 0.5)
            )
        return standard_route(messages)

    c = cfg()
    c.aggregate.plausible_threshold = 0.8
    r = run_pipeline(ws, RoutingLLM(route), c)
    assert [(f.file, f.verdict) for f in r.findings] == [("lib/util.py", "CONFIRMED")]
    assert "below_threshold" in [d.reason for d in r.dropped]


def test_rules_specialist_runs_when_rules_exist(ws) -> None:
    (ws.root / "AGENTS.md").write_text("Never change public function signatures.\n")
    (ws.root / "lib" / "AGENTS.md").write_text("lib: keep functions pure.\n")
    text = discover_rules(ws.guard, ["lib/util.py"], max_tokens=1_000)
    assert text.index("## AGENTS.md") < text.index("## lib/AGENTS.md")
    llm = RoutingLLM(standard_route)
    r = run_pipeline(ws, llm, cfg())
    assert r.stats["specialists"][-1] == "rules" and r.stats["rules_found"]
    assert discover_rules(ws.guard, ["lib/util.py"], max_tokens=5).endswith("[truncated]")


def test_review_workspace_dispatches_to_graph(ws) -> None:
    r = review_workspace(ws, RoutingLLM(standard_route), cfg())
    assert r.stats["pipeline"] == "graph"
    assert os.environ["LANGSMITH_TRACING"] == "false"  # forced off on import (PLAN §3)


def test_dedupe_keeps_distinct_neighbours_apart() -> None:
    def f(line, claim, cat="logic", conf=0.5):
        return Finding(
            id=f"{line}{claim}",
            file="x.py",
            line=line,
            category=cat,
            severity="major",
            claim=claim,
            failure_scenario="s",
            verdict="PLAUSIBLE",
            confidence=conf,
        )

    groups = dedupe(
        [
            (f(10, "null user dereferenced in login handler", conf=0.9), "logic"),
            (f(11, "login handler dereferences null user object", "contract"), "contract"),
            (f(12, "SQL query built by string concatenation", "security"), "security"),
            (f(40, "null user dereferenced in login handler"), "logic"),
        ]
    )
    assert [(g.finding.line, g.sources) for g in groups] == [
        (10, ["logic", "contract"]),
        (12, ["security"]),
        (40, ["logic"]),
    ]
    assert claim_similarity("a b c", "") == 0.0


def test_large_diffs_are_chunked() -> None:
    text = "".join(
        f"diff --git a/f{i}.py b/f{i}.py\n--- a/f{i}.py\n+++ b/f{i}.py\n@@ -0,0 +1,60 @@\n"
        + "+x = 1\n" * 60
        for i in range(3)
    )
    chunks, omitted = _chunk_files(parse_unified_diff(text), budget=300)
    assert [shown for _, shown in chunks] == [["f0.py"], ["f1.py"], ["f2.py"]] and not omitted


def test_eval_scores_before_and_after_verifier(tmp_path, upstream, monkeypatch) -> None:
    from deep_review import cli
    from deep_review.evaluation.dataset import EvalCase, GTIssue, save_cases
    from deep_review.llm import ScriptedLLM

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEP_REVIEW_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("DEEP_REVIEW_BASE_URL", "http://unused.invalid/v1")
    case = EvalCase(
        id="g1",
        source="t",
        repo_url=str(upstream["path"]),
        base_sha=upstream["main_commit"],
        head_sha=upstream["feature_commit"],
        language="Python",
        ground_truth=[
            GTIssue(
                id="g1#0",
                file="lib/util.py",
                line=1,
                end_line=2,
                text="signature change breaks callers",
                category="defect",
            )
        ],
    )
    ds = tmp_path / "ds.jsonl"
    save_cases([case], ds)
    assert (
        cli.main(
            ["eval", "run", "--dataset", str(ds), "--run-id", "g"], llm=RoutingLLM(standard_route)
        )
        == 0
    )
    judge = ScriptedLLM(
        [ChatResponse(message=Message.assistant('{"matches": [1], "reason": "same"}'))] * 3
    )
    assert cli.main(["eval", "score", "--dataset", str(ds), "--run-id", "g"], judge_llm=judge) == 0
    score = json.loads((tmp_path / "eval_runs" / "g" / "score.json").read_text())
    st = score["stages"]
    assert st["before_verifier"]["findings_per_case"] == 2.0  # util + new.py candidates
    assert st["after_verifier"]["findings_per_case"] == 1.0  # new.py rejected
    assert st["after_verifier"]["precision"] == 1.0 and st["before_verifier"]["precision"] == 0.5
    assert st["confirmed_only"]["recall"] == 1.0
    assert "Verifier effect" in (tmp_path / "eval_runs" / "g" / "report.md").read_text()


def test_dedupe_same_line_across_categories() -> None:
    def f(line, claim, cat):
        return Finding(
            id=f"{line}{cat}",
            file="t.py",
            line=line,
            category=cat,
            severity="major",
            claim=claim,
            failure_scenario="s",
            verdict="PLAUSIBLE",
            confidence=0.5,
        )

    a = (
        "Changing `if max_age is not None:` to `if max_age:` makes a max_age of 0 skip the "
        "expiry check entirely, so signatures that should be expired are accepted."
    )
    b = (
        "`if max_age:` treats `max_age=0` as 'no expiry check', so a zero max_age no longer "
        "rejects signatures whose age is non-zero."
    )
    assert len(dedupe([(f(138, a, "security"), "security"), (f(138, b, "logic"), "logic")])) == 1
    # same line, unrelated problems: stay apart
    c = "Cache key ignores the tenant id, so users can read each other's entries."
    assert len(dedupe([(f(138, a, "security"), "s"), (f(138, c, "logic"), "l")])) == 2
