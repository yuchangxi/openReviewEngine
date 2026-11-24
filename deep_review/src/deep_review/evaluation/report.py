"""Markdown reports: a run summary, a hand-check sample for the judge, and run comparison."""

from __future__ import annotations

import random
from typing import Any

from deep_review.evaluation.dataset import SUBSETS, EvalCase
from deep_review.evaluation.metrics import CaseScore


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def summary_markdown(summary: dict[str, Any], title: str) -> str:
    j, lw = summary["judge"], summary["line_window"]
    out = [f"# {title}", ""]
    out.append(
        f"Cases scored: {j['cases']} · missing/failed: {len(summary['missing'])} · "
        f"findings/case: {j['findings_per_case']}"
    )
    out += [
        "",
        "## Precision / recall (LLM judge)",
        "",
        "| Subset | Precision | Recall | F1 | TP findings / findings | covered / GT |",
        "|---|---|---|---|---|---|",
    ]
    for s in SUBSETS:
        m = j["overall"][s]
        out.append(
            f"| {s} | {_pct(m['precision'])} | {_pct(m['recall'])} | {_pct(m['f1'])} | "
            f"{m['tp_findings']} / {m['findings']} | {m['covered']} / {m['gt']} |"
        )
    m = lw["overall"]["bugs"]
    out += [
        "",
        f"Judge-free cross-check (same file, lines overlap ±{summary['line_window_size']}), "
        f"bugs subset: precision {_pct(m['precision'])}, recall {_pct(m['recall'])}.",
        "",
    ]
    if summary.get("stages"):
        out += [
            "## Verifier effect (bugs subset, same run)",
            "",
            "| Stage | Findings/case | Precision | Recall | F1 |",
            "|---|---|---|---|---|",
        ]
        for stage, m in summary["stages"].items():
            out.append(
                f"| {stage} | {m['findings_per_case']} | {_pct(m['precision'])} | "
                f"{_pct(m['recall'])} | {_pct(m['f1'])} |"
            )
        out.append("")
    if summary.get("threshold_curve"):
        out += [
            "## Confidence threshold sweep (bugs subset)",
            "",
            "| Min confidence | Findings/case | Precision | Recall | F1 |",
            "|---|---|---|---|---|",
        ]
        for t, m in summary["threshold_curve"].items():
            out.append(
                f"| ≥ {t} | {m['findings_per_case']} | {_pct(m['precision'])} | "
                f"{_pct(m['recall'])} | {_pct(m['f1'])} |"
            )
        out.append("")
    out += [
        "## By language (bugs subset)",
        "",
        "| Language | Cases | Precision | Recall | F1 |",
        "|---|---|---|---|---|",
    ]
    for lang, m in j["by_language"].items():
        out.append(
            f"| {lang} | {m['cases']} | {_pct(m['precision'])} | {_pct(m['recall'])} | "
            f"{_pct(m['f1'])} |"
        )
    out += [
        "",
        "## Finding precision by category",
        "",
        "| Category | Findings | TP | Precision |",
        "|---|---|---|---|",
    ]
    for cat, m in j["finding_precision_by_category"].items():
        out.append(f"| {cat} | {m['findings']} | {m['tp']} | {_pct(m['precision'])} |")
    c = j["cost"]
    out += [
        "",
        "## Cost",
        "",
        f"- Prompt tokens: {c['prompt_tokens']:,} (cached {c['cached_prompt_tokens']:,}); "
        f"completion tokens: {c['completion_tokens']:,}",
        f"- Turns: {c['turns']}, tool calls: {c['tool_calls']}",
        f"- Seconds per case: median {c['seconds_median']:.0f}, max {c['seconds_max']:.0f}",
        f"- Stop reasons: {j['stop_reasons']}",
        f"- Judge: {summary['judge_model']}, {summary['judge_calls']} calls, "
        f"{summary['judge_cache_hits']} cached, {summary.get('judge_errors', 0)} errors "
        "(errored findings count as unmatched; re-run score to retry them)",
    ]
    if summary["missing"]:
        out += ["", "## Missing or failed cases", ""] + [f"- {m}" for m in summary["missing"]]
    out += [
        "",
        "Caveat: ground truth is incomplete, so precision is a lower bound; "
        "spot-check `judge_sample.md` by hand before trusting the judge.",
    ]
    return "\n".join(out) + "\n"


def judge_sample_markdown(
    scores: list[CaseScore], cases: dict[str, EvalCase], n: int = 20, seed: int = 0
) -> str:
    """Random matched and unmatched findings with the references they were judged against."""
    rng = random.Random(seed)
    items = [(s, m) for s in scores for m in s.findings]
    hits = [x for x in items if x[1].matched]
    misses = [x for x in items if not x[1].matched]
    picked = rng.sample(hits, min(len(hits), n // 2)) + rng.sample(misses, min(len(misses), n // 2))
    out = [
        "# Judge spot-check sample",
        "",
        "Mark each verdict right or wrong by hand. Judge agreement below ~80% means the "
        "metrics above are not trustworthy.",
        "",
    ]
    for k, (s, m) in enumerate(picked, 1):
        gts = [g for g in cases[s.case_id].ground_truth if g.file in (m.file, "")]
        out += [
            f"## {k}. {s.case_id} — {'MATCH' if m.matched else 'NO MATCH'}",
            f"**Finding** `{m.file}:{m.line}` ({m.category}): {m.claim}",
            f"**Judge reason**: {m.reason}",
            "",
            "References in this file:",
        ]
        out += [
            f"- {'**✓**' if g.id in m.matched else '·'} `{g.line}-{g.end_line}` "
            f"[{g.category}] {g.text[:300]}"
            for g in gts
        ] or ["- (none)"]
        out += ["", "Verdict correct? [ ] yes  [ ] no", ""]
    return "\n".join(out)


def compare_markdown(base: dict[str, Any], new: dict[str, Any], names: tuple[str, str]) -> str:
    out = [
        f"# {names[1]} vs {names[0]}",
        "",
        "| Subset | Metric | Before | After | Δ |",
        "|---|---|---|---|---|",
    ]
    for s in SUBSETS:
        for metric in ("precision", "recall", "f1"):
            a = base["judge"]["overall"][s][metric]
            b = new["judge"]["overall"][s][metric]
            out.append(f"| {s} | {metric} | {_pct(a)} | {_pct(b)} | {(b - a) * 100:+.1f} pt |")
    ca, cb = base["judge"]["cost"], new["judge"]["cost"]
    out += [
        "",
        f"Prompt tokens: {ca['prompt_tokens']:,} → {cb['prompt_tokens']:,}; "
        f"median seconds/case: {ca['seconds_median']:.0f} → {cb['seconds_median']:.0f}",
        f"Cases: {base['judge']['cases']} → {new['judge']['cases']}",
    ]
    if base.get("dataset_sha256") != new.get("dataset_sha256"):
        out.append("\n**Warning: the two runs used different datasets.**")
    return "\n".join(out) + "\n"
