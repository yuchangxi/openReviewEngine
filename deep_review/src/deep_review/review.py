"""Review entry point. `review_workspace` dispatches on `cfg.pipeline.mode`:

- "graph" (M3, default): gate → context → specialists → dedupe → verifiers → aggregate
  (deep_review.pipeline).
- "single" (M1): one bounded agent. Its findings carry verdict PLAUSIBLE, unverified.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from deep_review import __version__
from deep_review.agent import SubmitTool, run_agent
from deep_review.config import Config
from deep_review.diff import FileDiff, select_files
from deep_review.diff.render import render_diff
from deep_review.findings import SEVERITY_RANK, Evidence, Finding, finding_id, position_ok
from deep_review.llm.tokens import estimate_tokens
from deep_review.llm.types import LLM, Message
from deep_review.prompts.reviewer import (
    SUBMIT_DESCRIPTION,
    SUBMIT_TOOL_NAME,
    Candidate,
    ReviewSubmission,
    build_system_prompt,
    build_user_message,
)
from deep_review.tools.repo_tools import ToolContext, default_tools
from deep_review.trace import Tracer
from deep_review.workspace import Workspace

ENGINE = "deep_review"


@dataclass(frozen=True)
class PRInfo:
    title: str = ""
    description: str = ""


class Dropped(BaseModel):
    reason: str
    file: str
    line: int
    claim: str


class ReviewResult(BaseModel):
    engine: str = ENGINE
    engine_version: str = __version__
    repo: str
    base_ref: str
    head_ref: str
    base_sha: str
    head_sha: str
    merge_base: str
    findings: list[Finding] = Field(default_factory=list)
    pre_verification: list[Finding] = Field(default_factory=list)
    """Deduplicated, position-checked candidates before the verifier ran (graph mode only),
    so one eval run can measure what verification adds or removes."""
    dropped: list[Dropped] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


def _normalize_path(path: str, known: set[str]) -> str:
    """Undo common model slips: './x', 'a/x', 'b/x', '/x'."""
    if path in known:
        return path
    p = path.strip().lstrip("/")
    for prefix in ("./", "a/", "b/"):
        if p.startswith(prefix) and p[len(prefix) :] in known:
            return p[len(prefix) :]
    return p.removeprefix("./")


def _to_finding(c: Candidate, known: set[str]) -> Finding:
    file = _normalize_path(c.file, known)
    return Finding(
        id=finding_id(ENGINE, file, c.line, c.claim),
        engine=ENGINE,
        file=file,
        line=c.line,
        end_line=c.end_line if c.end_line and c.end_line >= c.line else None,
        category=c.category,
        severity=c.severity,
        claim=c.claim.strip(),
        evidence=[
            Evidence(file=_normalize_path(e.file, known), line=e.line, note=e.note)
            for e in c.evidence
            if not PurePosixPath(e.file).is_absolute() and ".." not in PurePosixPath(e.file).parts
        ],
        failure_scenario=c.failure_scenario.strip(),
        suggestion=c.suggestion,
        verdict="PLAUSIBLE",
        confidence=c.confidence,
    )


def postprocess(
    candidates: list[Candidate],
    diffs: dict[str, FileDiff],
    *,
    max_comments: int,
    min_confidence: float = 0.0,
) -> tuple[list[Finding], list[Dropped]]:
    """Schema → position check → confidence threshold → dedupe → rank → cap (PLAN §4.5)."""
    known = set(diffs)
    kept: dict[str, Finding] = {}
    dropped: list[Dropped] = []
    for c in candidates:
        try:
            f = _to_finding(c, known)
        except ValidationError as e:
            dropped.append(
                Dropped(
                    reason=f"invalid: {e.errors()[0]['msg']}",
                    file=c.file,
                    line=c.line,
                    claim=c.claim,
                )
            )
            continue
        if not position_ok(f, diffs):
            dropped.append(Dropped(reason="outside_diff", file=f.file, line=f.line, claim=f.claim))
            continue
        if f.confidence < min_confidence:
            dropped.append(
                Dropped(reason="below_threshold", file=f.file, line=f.line, claim=f.claim)
            )
            continue
        if f.id in kept:
            dropped.append(Dropped(reason="duplicate", file=f.file, line=f.line, claim=f.claim))
            continue
        kept[f.id] = f
    ranked = sorted(kept.values(), key=lambda f: (SEVERITY_RANK[f.severity], -f.confidence))
    for f in ranked[max_comments:]:
        dropped.append(Dropped(reason="over_cap", file=f.file, line=f.line, claim=f.claim))
    return ranked[:max_comments], dropped


@dataclass
class Prepared:
    all_diffs: list[FileDiff]
    selected: list[FileDiff]
    skipped: list[tuple[FileDiff, Any]]
    result: ReviewResult
    stats: dict[str, Any]


def prepare(ws: Workspace, cfg: Config, tracer: Tracer) -> Prepared:
    """Diff, deterministic file selection, and the result skeleton (shared by both modes)."""
    all_diffs = ws.file_diffs()

    def read_head(path: str) -> str | None:
        try:
            return ws.guard.read_text(path, cfg.tools.max_file_bytes)
        except Exception:
            return None

    sel = select_files(
        all_diffs,
        read_head=read_head,
        max_changed_lines=cfg.selection.max_changed_lines_per_file,
        max_line_length=cfg.selection.max_line_length,
    )
    result = ReviewResult(
        repo=ws.repo_url,
        base_ref=ws.base_ref,
        head_ref=ws.head_ref,
        base_sha=ws.base_sha,
        head_sha=ws.head_sha,
        merge_base=ws.merge_base,
    )
    stats: dict[str, Any] = {
        "files_changed": len(all_diffs),
        "files_reviewed": len(sel.selected),
        "files_skipped": {fd.path: r.value for fd, r in sel.skipped},
    }
    tracer.event("selection", **stats)
    return Prepared(all_diffs, sel.selected, sel.skipped, result, stats)


def review_workspace(
    ws: Workspace,
    llm: LLM,
    cfg: Config,
    pr: PRInfo = PRInfo(),
    tracer: Tracer | None = None,
    verifier_llm: LLM | None = None,
) -> ReviewResult:
    tracer = tracer or Tracer.null()
    if cfg.pipeline.mode == "graph":
        from deep_review.pipeline import run_pipeline  # imported lazily: pulls in langgraph

        return run_pipeline(ws, llm, cfg, pr, tracer, verifier_llm=verifier_llm)
    return _review_single(ws, llm, cfg, pr, tracer)


def _review_single(
    ws: Workspace, llm: LLM, cfg: Config, pr: PRInfo, tracer: Tracer
) -> ReviewResult:
    t0 = time.perf_counter()
    prep = prepare(ws, cfg, tracer)
    all_diffs, result, stats = prep.all_diffs, prep.result, prep.stats
    sel_selected, sel_skipped = prep.selected, prep.skipped
    if not sel_selected:
        stats.update(stop_reason="nothing_to_review", duration_s=0.0)
        result.stats = stats
        return result

    # Leave room in the shared prefix for the system prompt and PR text.
    system_prompt = build_system_prompt(cfg.review.finder_mode)
    fixed = estimate_tokens(system_prompt + pr.title + pr.description) + 1_000
    rendered = render_diff(sel_selected, max(cfg.budgets.shared_prefix_tokens - fixed, 1_000))
    user = build_user_message(
        title=pr.title,
        description=pr.description,
        files=[(fd.path, fd.status, fd.additions, fd.deletions) for fd in sel_selected],
        skipped=[(fd.path, r.value) for fd, r in sel_skipped],
        diff_text=rendered.text,
        omitted=rendered.omitted,
        max_turns=cfg.budgets.max_turns,
    )
    messages = [Message.system(system_prompt), Message.user(user)]
    tracer.event(
        "prompt",
        system_chars=len(system_prompt),
        finder_mode=cfg.review.finder_mode,
        user=user,
        diff_tokens_est=rendered.tokens,
        omitted=rendered.omitted,
    )

    with tracer.span("agent", agent="single_reviewer"):
        agent = run_agent(
            llm,
            default_tools(),
            ToolContext.for_workspace(ws, cfg.tools),
            messages,
            SubmitTool(SUBMIT_TOOL_NAME, SUBMIT_DESCRIPTION, ReviewSubmission),
            max_turns=cfg.budgets.max_turns,
            context_budget_tokens=cfg.budgets.shared_prefix_tokens + cfg.budgets.tool_turn_tokens,
            max_output_tokens=cfg.llm.max_output_tokens,
            context_window_tokens=cfg.llm.context_tokens,
            tracer=tracer,
        )

    candidates = agent.submission.findings if isinstance(agent.submission, ReviewSubmission) else []
    findings, dropped = postprocess(
        candidates,
        {fd.path: fd for fd in all_diffs},
        max_comments=cfg.aggregate.max_comments_per_pr,
        min_confidence=cfg.aggregate.plausible_threshold,
    )
    result.findings, result.dropped = findings, dropped
    stats.update(
        stop_reason=agent.stop_reason,
        forced_finish=agent.forced_finish,
        error=agent.error,
        turns=agent.turns,
        tool_calls=agent.tool_calls,
        tool_errors=agent.tool_errors,
        candidates=len(candidates),
        prompt_tokens=agent.usage.prompt_tokens,
        completion_tokens=agent.usage.completion_tokens,
        cached_prompt_tokens=agent.usage.cached_prompt_tokens,
        peak_context_tokens=agent.peak_context_tokens,
        diff_files_omitted=rendered.omitted,
        duration_s=round(time.perf_counter() - t0, 1),
    )
    result.stats = stats
    tracer.event(
        "result",
        findings=len(findings),
        dropped=[d.model_dump() for d in dropped],
        **{k: v for k, v in stats.items() if k != "files_skipped"},
    )
    return result
