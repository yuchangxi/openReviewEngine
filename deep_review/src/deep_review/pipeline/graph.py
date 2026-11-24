"""M3 review pipeline on LangGraph (PLAN §4).

    build_context → specialists ×(N specialists × diff chunks)  (Send fan-out)
                  → dedupe (+ position check, verify cap)
                  → verify ×(candidates)                        (Send fan-out)
                  → aggregate (verdicts, threshold, rank, cap)

LangGraph only orchestrates: fan-out, reducers, concurrency limits, retries. Everything that
decides quality (agent loop, tools, prompts, dedupe, aggregation) is our own code (PLAN §3).
Runtime dependencies (workspace, models, config, tracer) travel in the LangGraph runtime
context, not in graph state, so state stays plain data.
"""

from __future__ import annotations

import os

# PLAN §3: LangSmith cloud tracing must be off. Force it, whatever the environment says,
# before langgraph/langsmith are imported.
for _var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
    os.environ[_var] = "false"

import operator  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Annotated, Any, TypedDict  # noqa: E402

from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.runtime import Runtime  # noqa: E402
from langgraph.types import RetryPolicy, Send  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from deep_review.agent import AgentResult, SubmitTool, run_agent  # noqa: E402
from deep_review.config import Config  # noqa: E402
from deep_review.diff import FileDiff  # noqa: E402
from deep_review.diff.render import render_diff, render_file  # noqa: E402
from deep_review.findings import SEVERITY_RANK, Evidence, Finding, position_ok  # noqa: E402
from deep_review.llm.tokens import estimate_tokens  # noqa: E402
from deep_review.llm.types import LLM, Message  # noqa: E402
from deep_review.pipeline.dedupe import dedupe  # noqa: E402
from deep_review.pipeline.rules import discover_rules  # noqa: E402
from deep_review.prompts.reviewer import (  # noqa: E402
    SUBMIT_DESCRIPTION,
    SUBMIT_TOOL_NAME,
    Candidate,
    ReviewSubmission,
    build_system_prompt,
    build_user_message,
)
from deep_review.prompts.specialists import SPECIALISTS, rules_specialist  # noqa: E402
from deep_review.prompts.verifier import (  # noqa: E402
    SUBMIT_VERDICT,
    VERDICT_DESCRIPTION,
    VERIFIER_RULES_SYSTEM,
    VERIFIER_SYSTEM,
    VerdictSubmission,
    build_verifier_message,
)
from deep_review.review import (  # noqa: E402
    Dropped,
    Prepared,
    PRInfo,
    ReviewResult,
    _normalize_path,
    _to_finding,
    prepare,
)
from deep_review.tools.repo_tools import ToolContext, default_tools  # noqa: E402
from deep_review.trace import Tracer  # noqa: E402
from deep_review.workspace import Workspace  # noqa: E402

MAX_CHUNKS = 4
UNVERIFIED_CONFIDENCE_CAP = 0.5


@dataclass
class Deps:
    ws: Workspace
    llm: LLM
    verifier_llm: LLM
    cfg: Config
    tracer: Tracer
    pr: PRInfo
    prepared: Prepared
    wide: dict[str, FileDiff] = field(default_factory=dict)
    """Selected files' diffs at the chosen (wider) context; set by build_context."""
    rules: str = ""
    """Repository rules text; set by build_context, shown to verifiers of rule findings."""


class State(TypedDict, total=False):
    system_prompt: str
    chunks: list[str]
    omitted: list[str]
    specialists: list[dict[str, str]]
    rules_found: bool
    context_lines: int | None
    candidates: Annotated[list[dict[str, Any]], operator.add]
    agent_runs: Annotated[list[dict[str, Any]], operator.add]
    to_verify: list[dict[str, Any]]
    pre_dropped: list[dict[str, Any]]
    verdicts: Annotated[list[dict[str, Any]], operator.add]
    final: dict[str, Any]


# ------------------------------------------------------------------ helpers


def _agent_stats(role: str, name: str, r: AgentResult) -> dict[str, Any]:
    return {
        "role": role,
        "name": name,
        "stop_reason": r.stop_reason,
        "forced_finish": r.forced_finish,
        "error": r.error,
        "turns": r.turns,
        "tool_calls": r.tool_calls,
        "tool_errors": r.tool_errors,
        "prompt_tokens": r.usage.prompt_tokens,
        "completion_tokens": r.usage.completion_tokens,
        "cached_prompt_tokens": r.usage.cached_prompt_tokens,
        "peak_context_tokens": r.peak_context_tokens,
    }


def _chunk_files(
    files: list[FileDiff], budget: int
) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Split the diff into chunks that each fit `budget`; most PRs fit in one."""
    chunks: list[tuple[str, list[str]]] = []
    remaining = list(files)
    while remaining and len(chunks) < MAX_CHUNKS:
        r = render_diff(remaining, budget)
        if not r.shown:  # a single file larger than the budget: show its beginning
            fd = min(remaining, key=lambda f: f.additions + f.deletions)
            text = render_file(fd)[: budget * 3] + "\n…[diff truncated; read the file with tools]"
            chunks.append((text, [fd.path]))
            remaining.remove(fd)
            continue
        chunks.append((r.text, r.shown))
        remaining = [f for f in remaining if f.path in set(r.omitted)]
    return chunks, [f.path for f in remaining]


def _report_text(f: Finding) -> str:
    lines = [
        f"File: {f.file}, line {f.line}" + (f"-{f.end_line}" if f.end_line else ""),
        f"Category: {f.category}; severity: {f.severity}",
        f"Claim: {f.claim}",
        f"Failure scenario: {f.failure_scenario}",
    ]
    if f.evidence:
        lines.append("Cited evidence:")
        lines += [f"- {e.file}:{e.line}: {e.note}" for e in f.evidence]
    return "\n".join(lines)


# ------------------------------------------------------------------ nodes


def _widest_fitting(d: Deps, budget: int) -> tuple[list[FileDiff], int | None]:
    """Selected files rendered with the widest context that fits the budget in one chunk.
    Positions are still checked against the ±3-line diff in `prepared`, never the wide one."""
    selected = [f.path for f in d.prepared.selected]
    for lines in d.cfg.pipeline.context_lines:
        if lines == 3:
            wide = {f.path: f for f in d.prepared.selected}
        else:
            parsed = {f.path: f for f in d.ws.file_diffs(context_lines=lines)}
            wide = {p: parsed.get(p) for p in selected}
            if any(v is None for v in wide.values()):
                continue  # rename detection differed at this width; try the next one
        files = [wide[p] for p in selected]
        if not render_diff(files, budget).omitted:
            d.wide = wide  # type: ignore[assignment]
            return files, lines
    d.wide = {f.path: f for f in d.prepared.selected}
    return list(d.prepared.selected), None


def _warm_prefix(d: Deps, system_prompt: str, user: str) -> None:
    """Populate the prompt cache with the shared prefix (same messages and tools as the
    specialists send) before they start in parallel. Failure is harmless."""
    specs = [t.spec for t in default_tools()] + [
        SubmitTool(SUBMIT_TOOL_NAME, SUBMIT_DESCRIPTION, ReviewSubmission).spec
    ]
    try:
        resp = d.llm.chat(
            [Message.system(system_prompt), Message.user(user)], tools=specs, max_tokens=16
        )
        d.tracer.event("warm_prefix", usage=resp.usage.model_dump())
    except Exception as e:  # noqa: BLE001 - warming is an optimisation only
        d.tracer.event("warm_prefix_failed", error=str(e))


def build_context(state: State, runtime: Runtime[Deps]) -> State:
    d = runtime.context
    cfg, prep = d.cfg, d.prepared
    system_prompt = build_system_prompt(cfg.review.finder_mode)
    rules = discover_rules(
        d.ws.guard, [f.path for f in prep.selected], cfg.pipeline.max_rules_tokens
    )
    fixed = estimate_tokens(system_prompt + d.pr.title + d.pr.description + rules) + 1_500
    budget = max(cfg.budgets.shared_prefix_tokens - fixed, 2_000)
    files_for_render, context_lines = _widest_fitting(d, budget)
    chunks, omitted = _chunk_files(files_for_render, budget)
    files = [(f.path, f.status, f.additions, f.deletions) for f in prep.selected]
    skipped = [(f.path, r.value) for f, r in prep.skipped]
    users = [
        build_user_message(
            title=d.pr.title,
            description=d.pr.description,
            files=files,
            skipped=skipped,
            diff_text=text,
            omitted=[p for p in (f.path for f in prep.selected) if p not in set(shown)],
            max_turns=cfg.pipeline.specialist_max_turns,
            context_lines=context_lines,
        )
        for text, shown in chunks
    ]
    if cfg.pipeline.warm_prefix and users:
        _warm_prefix(d, system_prompt, users[0])
    specs = [
        {"name": SPECIALISTS[n].name, "focus": SPECIALISTS[n].focus}
        for n in cfg.pipeline.specialists
    ]
    d.rules = rules
    if rules:
        rs = rules_specialist(rules)
        specs.append({"name": rs.name, "focus": rs.focus})
    d.tracer.event(
        "context",
        chunks=len(users),
        context_lines=context_lines,
        omitted=omitted,
        specialists=[s["name"] for s in specs],
        rules_chars=len(rules),
        user=users[0] if users else "",
    )
    return {
        "system_prompt": system_prompt,
        "chunks": users,
        "omitted": omitted,
        "specialists": specs,
        "rules_found": bool(rules),
        "context_lines": context_lines,
    }


def fan_out_specialists(state: State) -> list[Send] | str:
    sends = [
        Send(
            "specialist",
            {
                "name": s["name"],
                "focus": s["focus"],
                "chunk": i,
                "user": user,
                "system": state["system_prompt"],
            },
        )
        for i, user in enumerate(state.get("chunks", []))
        for s in state.get("specialists", [])
    ]
    return sends or "aggregate"


def specialist(payload: dict[str, Any], runtime: Runtime[Deps]) -> State:
    d = runtime.context
    cfg = d.cfg
    source = f"{payload['name']}#{payload['chunk']}"
    # Shared prefix (system + PR/diff) first, the specialist's focus last: prefix-cache friendly.
    messages = [
        Message.system(payload["system"]),
        Message.user(payload["user"]),
        Message.user(payload["focus"]),
    ]
    with d.tracer.span("specialist", specialist=source):
        r = run_agent(
            d.llm,
            default_tools(),
            ToolContext.for_workspace(d.ws, cfg.tools),
            messages,
            SubmitTool(SUBMIT_TOOL_NAME, SUBMIT_DESCRIPTION, ReviewSubmission),
            max_turns=cfg.pipeline.specialist_max_turns,
            context_budget_tokens=cfg.budgets.shared_prefix_tokens + cfg.budgets.tool_turn_tokens,
            max_output_tokens=cfg.llm.max_output_tokens,
            context_window_tokens=cfg.llm.context_tokens,
            tracer=d.tracer,
        )
    found = r.submission.findings if isinstance(r.submission, ReviewSubmission) else []
    return {
        "candidates": [{"source": source, "candidate": c.model_dump()} for c in found],
        "agent_runs": [_agent_stats("specialist", source, r) | {"candidates": len(found)}],
    }


def dedupe_node(state: State, runtime: Runtime[Deps]) -> State:
    d = runtime.context
    diffs = {f.path: f for f in d.prepared.all_diffs}
    known = set(diffs)
    items: list[tuple[Finding, str]] = []
    dropped: list[dict[str, Any]] = []
    for rec in state.get("candidates", []):
        c = Candidate.model_validate(rec["candidate"])
        try:
            f = _to_finding(c, known)
        except ValidationError as e:
            dropped.append(
                Dropped(
                    reason=f"invalid: {e.errors()[0]['msg']}",
                    file=c.file,
                    line=c.line,
                    claim=c.claim,
                ).model_dump()
            )
            continue
        if not position_ok(f, diffs):
            dropped.append(
                Dropped(reason="outside_diff", file=f.file, line=f.line, claim=f.claim).model_dump()
            )
            continue
        items.append((f, rec["source"]))
    groups = dedupe(items, window=d.cfg.pipeline.dedupe_line_window)
    for g in groups:
        for m in g.merged:
            dropped.append(
                Dropped(
                    reason=f"merged_into:{g.finding.id}", file=m.file, line=m.line, claim=m.claim
                ).model_dump()
            )
    groups.sort(key=lambda g: (SEVERITY_RANK[g.finding.severity], -g.finding.confidence))
    cap = d.cfg.pipeline.max_candidates_to_verify
    for g in groups[cap:]:
        dropped.append(
            Dropped(
                reason="verify_cap", file=g.finding.file, line=g.finding.line, claim=g.finding.claim
            ).model_dump()
        )
    to_verify = [{"finding": g.finding.model_dump(), "sources": g.sources} for g in groups[:cap]]
    d.tracer.event("dedupe", candidates=len(items), groups=len(groups), to_verify=len(to_verify))
    return {"to_verify": to_verify, "pre_dropped": dropped}


def _agreed(item: dict[str, Any], cfg: Config) -> bool:
    return len(item.get("sources", [])) >= cfg.pipeline.min_agreement


def _needs_verifier(item: dict[str, Any], cfg: Config) -> bool:
    mode = cfg.pipeline.filter
    return mode == "verifier" or (mode == "agreement_or_verifier" and not _agreed(item, cfg))


def fan_out_verify(state: State, runtime: Runtime[Deps]) -> list[Send] | str:
    cfg = runtime.context.cfg
    sends = [Send("verify", it) for it in state.get("to_verify", []) if _needs_verifier(it, cfg)]
    return sends or "aggregate"


def verify(payload: dict[str, Any], runtime: Runtime[Deps]) -> State:
    d = runtime.context
    cfg = d.cfg
    f = Finding.model_validate(payload["finding"])
    fd = d.wide.get(f.file) or next((x for x in d.prepared.all_diffs if x.path == f.file), None)
    is_rule = bool(d.rules) and any(s.startswith("rules#") for s in payload.get("sources", []))
    user = build_verifier_message(
        title=d.pr.title,
        description=d.pr.description,
        file_diff=render_file(fd)[: cfg.budgets.shared_prefix_tokens * 3] if fd else "",
        report=_report_text(f),
        max_turns=cfg.pipeline.verifier_max_turns,
        rules=d.rules if is_rule else "",
    )
    system = VERIFIER_RULES_SYSTEM if is_rule else VERIFIER_SYSTEM
    with d.tracer.span("verifier", finding=f.id, rules=is_rule):
        r = run_agent(
            d.verifier_llm,
            default_tools(),
            ToolContext.for_workspace(d.ws, cfg.tools),
            [Message.system(system), Message.user(user)],
            SubmitTool(SUBMIT_VERDICT, VERDICT_DESCRIPTION, VerdictSubmission),
            max_turns=cfg.pipeline.verifier_max_turns,
            context_budget_tokens=cfg.budgets.shared_prefix_tokens + cfg.budgets.tool_turn_tokens,
            max_output_tokens=cfg.llm.max_output_tokens,
            context_window_tokens=cfg.llm.context_tokens,
            tracer=d.tracer,
        )
    v = r.submission if isinstance(r.submission, VerdictSubmission) else None
    d.tracer.event("verdict", finding=f.id, verdict=v.model_dump() if v else None)
    return {
        "verdicts": [{"finding_id": f.id, "verdict": v.model_dump() if v else None}],
        "agent_runs": [_agent_stats("verifier", f.id, r)],
    }


def aggregate(state: State, runtime: Runtime[Deps]) -> State:
    """Verdicts → threshold → rank → cap (PLAN §4.5)."""
    d = runtime.context
    cfg = d.cfg
    known = {f.path for f in d.prepared.all_diffs}
    verdicts = {v["finding_id"]: v["verdict"] for v in state.get("verdicts", [])}
    mode = cfg.pipeline.filter
    pre: list[dict[str, Any]] = []
    kept: dict[str, Finding] = {}
    dropped = list(state.get("pre_dropped", []))
    counts: Counter[str] = Counter()
    for item in state.get("to_verify", []):
        f = Finding.model_validate(item["finding"])
        pre.append(f.model_dump())
        accepted = False  # accepted by agreement: exempt from the confidence threshold
        if mode == "none":
            out = f
        elif mode in ("agreement", "agreement_or_verifier") and _agreed(item, cfg):
            counts["agreement"] += 1
            out, accepted = f, True
        elif mode == "agreement":
            counts["single_source"] += 1
            dropped.append(
                Dropped(
                    reason="single_source", file=f.file, line=f.line, claim=f.claim
                ).model_dump()
            )
            continue
        else:
            v = verdicts.get(f.id)
            if v is None:  # verifier gave no verdict: keep, but never above the cap
                counts["no_verdict"] += 1
                out = f.model_copy(
                    update={"confidence": min(f.confidence, UNVERIFIED_CONFIDENCE_CAP)}
                )
            elif v["verdict"] == "REJECTED":
                counts["REJECTED"] += 1
                dropped.append(
                    Dropped(
                        reason=f"rejected: {v['reason'][:300]}",
                        file=f.file,
                        line=f.line,
                        claim=f.claim,
                    ).model_dump()
                )
                continue
            else:
                counts[v["verdict"]] += 1
                extra = [
                    Evidence(file=_normalize_path(e["file"], known), line=e["line"], note=e["note"])
                    for e in v.get("evidence", [])
                    if not e["file"].startswith("/") and ".." not in e["file"].split("/")
                ]
                out = f.model_copy(
                    update={
                        "verdict": v["verdict"],
                        "confidence": v["confidence"],
                        "evidence": (f.evidence + extra)[:8],
                    }
                )
        if (
            not accepted
            and out.verdict != "CONFIRMED"
            and out.confidence < cfg.aggregate.plausible_threshold
        ):
            dropped.append(
                Dropped(
                    reason="below_threshold", file=out.file, line=out.line, claim=out.claim
                ).model_dump()
            )
            continue
        if out.id in kept:
            continue
        kept[out.id] = out
    ranked = sorted(
        kept.values(),
        key=lambda f: (f.verdict != "CONFIRMED", SEVERITY_RANK[f.severity], -f.confidence),
    )
    cap = cfg.aggregate.max_comments_per_pr
    for f in ranked[cap:]:
        dropped.append(
            Dropped(reason="over_cap", file=f.file, line=f.line, claim=f.claim).model_dump()
        )
    return {
        "final": {
            "findings": [f.model_dump() for f in ranked[:cap]],
            "pre_verification": pre,
            "dropped": dropped,
            "verdict_counts": dict(counts),
        }
    }


# ------------------------------------------------------------------ graph


def build_graph():
    retry = RetryPolicy(max_attempts=2)
    g = StateGraph(State, context_schema=Deps)
    g.add_node("build_context", build_context)
    g.add_node("specialist", specialist, retry_policy=retry)
    g.add_node("dedupe", dedupe_node)
    g.add_node("verify", verify, retry_policy=retry)
    g.add_node("aggregate", aggregate)
    g.add_edge(START, "build_context")
    g.add_conditional_edges("build_context", fan_out_specialists, ["specialist", "aggregate"])
    g.add_edge("specialist", "dedupe")
    g.add_conditional_edges("dedupe", fan_out_verify, ["verify", "aggregate"])
    g.add_edge("verify", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()


def run_pipeline(
    ws: Workspace,
    llm: LLM,
    cfg: Config,
    pr: PRInfo | None = None,
    tracer: Tracer | None = None,
    *,
    verifier_llm: LLM | None = None,
) -> ReviewResult:
    pr = pr or PRInfo()
    tracer = tracer or Tracer.null()
    t0 = time.perf_counter()
    prep = prepare(ws, cfg, tracer)
    result, stats = prep.result, prep.stats
    if not prep.selected:  # gate: nothing reviewable
        stats.update(stop_reason="nothing_to_review", duration_s=0.0)
        result.stats = stats
        return result

    deps = Deps(ws, llm, verifier_llm or llm, cfg, tracer, pr, prep)
    with tracer.span("pipeline"):
        state = build_graph().invoke(
            {}, config={"max_concurrency": cfg.pipeline.max_concurrency}, context=deps
        )
    final = state["final"]
    result.findings = [Finding.model_validate(f) for f in final["findings"]]
    result.pre_verification = [Finding.model_validate(f) for f in final["pre_verification"]]
    result.dropped = [Dropped.model_validate(x) for x in final["dropped"]]

    runs = state.get("agent_runs", [])

    def total(key: str, role: str | None = None) -> int:
        return sum(int(r.get(key) or 0) for r in runs if role is None or r["role"] == role)

    stats.update(
        pipeline="graph",
        stop_reason="completed",
        specialists=[s["name"] for s in state.get("specialists", [])],
        diff_chunks=len(state.get("chunks", [])),
        context_lines=state.get("context_lines"),
        diff_files_omitted=state.get("omitted", []),
        rules_found=state.get("rules_found", False),
        candidates=total("candidates", "specialist"),
        verified=len(final["pre_verification"]),
        verdict_counts=final["verdict_counts"],
        agent_stop_reasons=dict(Counter(f"{r['role']}:{r['stop_reason']}" for r in runs)),
        turns=total("turns"),
        tool_calls=total("tool_calls"),
        tool_errors=total("tool_errors"),
        prompt_tokens=total("prompt_tokens"),
        completion_tokens=total("completion_tokens"),
        cached_prompt_tokens=total("cached_prompt_tokens"),
        specialist_prompt_tokens=total("prompt_tokens", "specialist"),
        verifier_prompt_tokens=total("prompt_tokens", "verifier"),
        peak_context_tokens=max((r["peak_context_tokens"] for r in runs), default=0),
        agents=runs,
        finding_sources={
            it["finding"]["id"]: [x.split("#")[0] for x in it.get("sources", [])]
            for it in state.get("to_verify", [])
        },
        duration_s=round(time.perf_counter() - t0, 1),
    )
    result.stats = stats
    tracer.event(
        "result",
        findings=len(result.findings),
        pre_verification=len(result.pre_verification),
        **{k: v for k, v in stats.items() if k not in ("files_skipped", "agents")},
    )
    return result
