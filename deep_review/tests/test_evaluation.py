from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_review import cli
from deep_review.evaluation.aacr import convert_record, import_aacr
from deep_review.evaluation.dataset import EvalCase, GTIssue, load_cases, save_cases
from deep_review.evaluation.judge import Judge, line_window_matches
from deep_review.evaluation.metrics import aggregate, score_case
from deep_review.evaluation.qodo import convert_record as convert_qodo
from deep_review.evaluation.qodo import import_qodo
from deep_review.findings import Finding
from deep_review.llm import ChatResponse, Message, ScriptedLLM, ToolCall
from deep_review.prompts.reviewer import SUBMIT_TOOL_NAME
from deep_review.review import ReviewResult

AACR_RECORD = {
    "category": "Bug Fix",
    "change_line_count": "120",
    "githubPrUrl": "https://github.com/acme/bank-core/pull/42",
    "project_main_language": "Java",
    "source_commit": "a" * 40,
    "target_commit": "b" * 40,
    "comments": [
        {
            "category": "Code Defect",
            "context": "Diff Level",
            "from_line": 12,
            "to_line": 10,
            "is_ai_comment": False,
            "note": "Uses double for money.",
            "path": "src/Pay.java",
            "side": "right",
            "source_model": "",
        },
        {
            "category": "Maintainability and Readability",
            "context": "File Level",
            "from_line": 3,
            "to_line": 3,
            "is_ai_comment": True,
            "note": "Rename x.",
            "path": "/src/Pay.java",
            "side": "right",
            "source_model": "m",
        },
        {
            "category": "Code Defect",
            "from_line": 1,
            "to_line": 1,
            "note": "old side",
            "path": "src/Pay.java",
            "side": "left",
        },
        {
            "category": "Performance",
            "from_line": 5,
            "to_line": 6,
            "note": "  ",
            "path": "src/Pay.java",
            "side": "right",
        },
    ],
}


def gt(i: int, file: str, line: int, cat: str = "defect", end: int | None = None) -> GTIssue:
    return GTIssue(
        id=f"c#{i}", file=file, line=line, end_line=end or line, text=f"issue {i}", category=cat
    )


def finding(file: str, line: int, claim: str = "bug") -> Finding:
    return Finding(
        id=f"{file}:{line}:{claim}",
        file=file,
        line=line,
        category="logic",
        severity="major",
        claim=claim,
        failure_scenario="x",
        verdict="PLAUSIBLE",
        confidence=0.8,
    )


def verdict(*nums: int) -> ChatResponse:
    return ChatResponse(
        message=Message.assistant(json.dumps({"matches": list(nums), "reason": "r"}))
    )


def test_convert_aacr_record() -> None:
    case = convert_record(AACR_RECORD)
    assert case.id == "acme__bank-core-42"
    assert case.repo_url == "https://github.com/acme/bank-core.git"
    assert (case.base_sha, case.head_sha) == ("a" * 40, "b" * 40)
    assert case.language == "Java" and case.changed_lines == 120
    # left-side and empty comments dropped; reversed range fixed; leading "/" stripped
    assert [(g.file, g.line, g.end_line, g.category, g.origin) for g in case.ground_truth] == [
        ("src/Pay.java", 10, 12, "defect", "human"),
        ("src/Pay.java", 3, 3, "maintainability", "ai"),
    ]


def test_import_filters_languages_and_survives_meta_errors() -> None:
    other = AACR_RECORD | {
        "githubPrUrl": "https://github.com/x/y/pull/1",
        "project_main_language": "Rust",
    }
    calls = []

    def fake_fetch(url, token):
        calls.append(url)
        raise ValueError("rate limited")

    logs = []
    cases = import_aacr([AACR_RECORD, other], languages={"Java"}, fetch=fake_fetch, log=logs.append)
    assert [c.id for c in cases] == ["acme__bank-core-42"] and calls == [AACR_RECORD["githubPrUrl"]]
    assert "rate limited" in logs[0] and cases[0].title == ""


def test_dataset_round_trip_rejects_duplicates(tmp_path: Path) -> None:
    case = convert_record(AACR_RECORD)
    p = tmp_path / "d.jsonl"
    save_cases([case], p)
    assert load_cases(p) == [case]
    save_cases([case, case], p)
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(p)


def test_judge_maps_numbers_caches_and_retries(tmp_path: Path) -> None:
    refs = [gt(1, "f.py", 10), gt(2, "f.py", 50)]
    llm = ScriptedLLM(
        [
            ChatResponse(message=Message.assistant("not json")),  # retried
            verdict(2, 7),  # 7 is out of range → ignored
        ]
    )
    judge = Judge(llm, tmp_path / "cache.jsonl", model_name="judge-m")
    assert judge.match(finding("f.py", 48), refs) == (["c#2"], "r")
    assert judge.calls == 2
    # same question again → cache, even from a fresh Judge reading the file
    judge2 = Judge(ScriptedLLM([]), tmp_path / "cache.jsonl", model_name="judge-m")
    assert judge2.match(finding("f.py", 48), refs)[0] == ["c#2"] and judge2.cache_hits == 1
    # no references in the file → no model call at all
    assert judge2.match(finding("g.py", 1), []) == ([], "no reference in this file")
    flaky = ScriptedLLM(["x", "y", "z", verdict(1)])
    j3 = Judge(flaky, tmp_path / "c3.jsonl")
    ids, reason = j3.match(finding("f.py", 1), refs)
    assert ids == [] and reason.startswith("JUDGE_ERROR") and j3.errors == 1
    assert [c["max_tokens"] for c in flaky.calls] == [8_000, 16_000, 32_000]
    assert j3.match(finding("f.py", 1), refs)[0] == ["c#1"]  # errors are not cached → retried


def test_line_window() -> None:
    refs = [gt(1, "f.py", 10, end=12), gt(2, "f.py", 30)]
    assert line_window_matches(finding("f.py", 14), refs, window=3) == ["c#1"]
    assert line_window_matches(finding("f.py", 16), refs, window=3) == []


def test_scoring_is_greedy_and_counts_subsets() -> None:
    case = EvalCase(
        id="c",
        source="t",
        repo_url="r",
        base_sha="a",
        head_sha="b",
        language="Java",
        ground_truth=[
            gt(1, "f.py", 10),
            gt(2, "f.py", 20, "maintainability"),
            gt(3, "g.py", 5, "security"),
        ],
    )
    result = ReviewResult(
        repo="r",
        base_ref="a",
        head_ref="b",
        base_sha="a",
        head_sha="b",
        merge_base="a",
        findings=[
            finding("f.py", 10, "first"),
            finding("f.py", 11, "dup"),
            finding("f.py", 20, "style"),
            finding("h.py", 1, "noise"),
        ],
        stats={"prompt_tokens": 100, "duration_s": 3.0, "stop_reason": "submitted"},
    )
    answers = {"first": ["c#1"], "dup": ["c#1"], "style": ["c#2"]}
    s = score_case(case, result, lambda f, refs: (answers.get(f.claim, []), ""))
    assert [m.tp for m in s.findings] == [True, False, True, False]
    assert s.gt_total == {"all": 3, "bugs": 2, "defect_security": 2}
    assert s.gt_covered == {"all": 2, "bugs": 1, "defect_security": 1}
    agg = aggregate([s])
    assert agg["overall"]["all"] == {
        "precision": 0.5,
        "recall": 0.667,
        "f1": 0.571,
        "tp_findings": 2,
        "findings": 4,
        "covered": 2,
        "gt": 3,
    }
    assert agg["overall"]["bugs"]["recall"] == 0.5
    assert agg["by_language"]["Java"]["cases"] == 1
    assert agg["cost"]["prompt_tokens"] == 100


QODO_URL = "https://github.com/agentic-review-benchmarks/Ghost/pull/4"
QODO_RECORD = {
    "repo": "Ghost",
    "pr_url_to_review": QODO_URL,
    "num_of_issues": 3,
    "issues": [
        {
            "title": "Missing semicolons",
            "description": "Rule 3 requires semicolons.",
            "file_path": "ghost/core/a.js",
            "start_line": 12,
            "end_line": 10,
            "problematic_code_snippet": "let a = 1",
            "rule_name": "Use Semicolons",
        },
        {
            "title": "Delete removed",
            "description": "Users can no longer delete comments.",
            "file_path": "None",
            "start_line": "None",
            "end_line": "None",
            "problematic_code_snippet": "<DropdownMenu>",
        },
        {
            "title": "Off by one",
            "description": "d",
            "file_path": "/b.js",
            "start_line": 5,
            "end_line": 7,
            "rule_name": None,
        },
    ],
}
QODO_REFS = {
    "base_sha": "a" * 40,
    "head_sha": "b" * 40,
    "title": "Code Review Bench PR #25799 - Remove delete",
    "body": "x" * 5000,
}


def test_convert_qodo_record() -> None:
    c = convert_qodo(QODO_RECORD, QODO_REFS)
    assert (c.id, c.repo_url, c.language) == (
        "qodo__Ghost-4",
        "https://github.com/agentic-review-benchmarks/Ghost.git",
        "JavaScript",
    )
    assert (c.base_sha, c.head_sha, c.title) == ("a" * 40, "b" * 40, "Remove delete")
    assert c.description.endswith("[truncated]")
    rule, unlocated, bug = c.ground_truth
    assert (rule.file, rule.line, rule.end_line, rule.category) == (
        "ghost/core/a.js",
        12,
        12,
        "maintainability",
    )
    assert rule.scope == "rule: Use Semicolons" and "Code: let a = 1" in rule.text
    assert (unlocated.file, unlocated.line, unlocated.category) == ("", 1, "defect")
    assert (bug.file, bug.line, bug.end_line) == ("b.js", 5, 7)
    with pytest.raises(ValueError, match="no base/head refs"):
        import_qodo([QODO_RECORD], refs={})


def test_unlocated_ground_truth_is_offered_for_any_file_but_not_line_window() -> None:
    case = EvalCase(
        id="c",
        source="t",
        repo_url="r",
        base_sha="a",
        head_sha="b",
        ground_truth=[gt(1, "f.py", 10), gt(2, "", 1)],
    )
    result = ReviewResult(
        repo="r",
        base_ref="a",
        head_ref="b",
        base_sha="a",
        head_sha="b",
        merge_base="a",
        findings=[finding("other.py", 1, "x")],
    )
    seen: list[list[str]] = []
    score_case(case, result, lambda f, refs: (seen.append([g.id for g in refs]) or [], ""))
    assert seen == [["c#2"]]
    assert line_window_matches(finding("other.py", 1), [gt(2, "", 1)]) == []


def _submit(findings: list[dict]) -> Message:
    return Message.assistant(
        None,
        [ToolCall(id="s", name=SUBMIT_TOOL_NAME, arguments=json.dumps({"findings": findings}))],
    )


def test_eval_cli_run_score_compare(tmp_path: Path, upstream, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # no .env; runs land in ./eval_runs
    monkeypatch.setenv("DEEP_REVIEW_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("DEEP_REVIEW_BASE_URL", "http://unused.invalid/v1")
    monkeypatch.setenv("DEEP_REVIEW_PIPELINE", "single")
    good = EvalCase(
        id="local-1",
        source="test",
        repo_url=str(upstream["path"]),
        base_sha=upstream["main_commit"],
        head_sha=upstream["feature_commit"],
        language="Python",
        ground_truth=[gt(1, "lib/util.py", 1, end=2), gt(2, "new.py", 3, "maintainability")],
    )
    broken = good.model_copy(update={"id": "local-broken", "head_sha": "c" * 40})
    dataset = tmp_path / "ds.jsonl"
    save_cases([good, broken], dataset)

    engine = ScriptedLLM(
        [
            _submit(
                [
                    {
                        "file": "lib/util.py",
                        "line": 1,
                        "category": "contract",
                        "severity": "major",
                        "claim": "signature changed",
                        "failure_scenario": "old callers",
                        "confidence": 0.9,
                    },
                    {
                        "file": "new.py",
                        "line": 1,
                        "category": "logic",
                        "severity": "minor",
                        "claim": "unrelated",
                        "failure_scenario": "x",
                        "confidence": 0.7,
                    },
                ]
            )
        ]
    )
    argv = ["eval", "run", "--dataset", str(dataset), "--run-id", "base", "--parallel", "1"]
    assert cli.main(argv, llm=engine) == 0
    run_dir = tmp_path / "eval_runs" / "base"
    assert (run_dir / "cases" / "local-1.json").exists()
    err = (run_dir / "cases" / "local-broken.error").read_text()
    assert "GitError" in err and not (run_dir / "cases" / "local-broken.json").exists()
    assert (run_dir / "recordings" / "local-1.jsonl").exists()
    # resume: the finished case is skipped, so no model call is needed
    assert cli.main(argv, llm=ScriptedLLM([])) == 0
    # a partial re-run must not shrink the manifest's case list
    assert cli.main([*argv, "--cases", "local-1"], llm=ScriptedLLM([])) == 0
    assert json.loads((run_dir / "run.json").read_text())["case_ids"] == ["local-1", "local-broken"]

    judge = ScriptedLLM([verdict(1), verdict()])
    assert (
        cli.main(
            ["eval", "score", "--dataset", str(dataset), "--run-id", "base", "--parallel", "1"],
            judge_llm=judge,
        )
        == 0
    )
    score = json.loads((run_dir / "score.json").read_text())
    assert score["missing"] == ["local-broken"]
    assert score["judge"]["overall"]["all"]["precision"] == 0.5
    assert score["judge"]["overall"]["bugs"]["recall"] == 1.0
    assert score["line_window"]["overall"]["all"]["tp_findings"] == 2  # both within ±3 lines
    assert "Precision / recall" in (run_dir / "report.md").read_text()
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["config"]["aggregate"]["plausible_threshold"] == 0.0  # keep all findings
    curve = score["threshold_curve"]
    assert list(curve) == ["0.0", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    assert curve["0.0"]["findings_per_case"] == 2.0  # both findings (conf 0.9 and 0.7)
    assert curve["0.8"]["findings_per_case"] == 1.0 and curve["0.8"]["precision"] == 1.0
    assert "threshold sweep" in (run_dir / "report.md").read_text()
    assert "Judge spot-check" in (run_dir / "judge_sample.md").read_text()

    # re-scoring uses the judge cache: no model calls
    assert (
        cli.main(
            ["eval", "score", "--dataset", str(dataset), "--run-id", "base"],
            judge_llm=ScriptedLLM([]),
        )
        == 0
    )

    (tmp_path / "eval_runs" / "after").mkdir()
    (tmp_path / "eval_runs" / "after" / "score.json").write_text(json.dumps(score))
    assert cli.main(["eval", "compare", "base", "after"]) == 0


def test_llm_errors_fail_the_case_and_rate_limits_stop_the_run(tmp_path: Path, monkeypatch) -> None:
    import contextlib

    from deep_review.config import load_config
    from deep_review.evaluation import runner

    class FakeMgr:
        def sweep_stale(self) -> list[str]:
            return []

        @contextlib.contextmanager
        def checkout(self, *a, **k):
            yield None

    monkeypatch.setattr(runner.WorkspaceManager, "from_config", lambda cfg: FakeMgr())
    reviewed: list[str] = []

    def fake_review(ws, llm, cfg, pr, tracer, verifier_llm=None):
        reviewed.append(pr.title)
        err = "chat completion failed (RateLimitError): 429 usage limit reached"
        return ReviewResult(
            repo="r",
            base_ref="a",
            head_ref="b",
            base_sha="a",
            head_sha="b",
            merge_base="a",
            stats={"agents": [{"stop_reason": "llm_error", "error": err}]},
        )

    monkeypatch.setattr(runner, "review_workspace", fake_review)
    cases = [
        EvalCase(id=f"c{i}", source="t", repo_url="r", base_sha="a", head_sha="b", title=f"c{i}")
        for i in range(3)
    ]
    dataset = tmp_path / "ds.jsonl"
    save_cases(cases, dataset)
    out = runner.run_dataset(
        cases,
        dataset,
        tmp_path / "run",
        load_config(env={}),
        ScriptedLLM([]),
        parallel=1,
        record=False,
    )
    assert [o.status for o in out] == ["failed", "aborted", "aborted"]
    assert reviewed == ["c0"]  # quota gone: no further cases were spent
    assert not (tmp_path / "run" / "cases" / "c0.json").exists()  # incomplete → re-run later
    assert "RateLimitError" in (tmp_path / "run" / "cases" / "c0.error").read_text()
