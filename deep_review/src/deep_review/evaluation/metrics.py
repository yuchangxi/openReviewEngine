"""Scoring: precision / recall / F1 per subset, language and category, plus cost.

Counting (as in AACR-Bench's coverage metric):
- A finding is a true positive if it covers at least one ground-truth issue that no earlier
  finding (in the engine's own rank order) has covered. Re-covering is a duplicate → not a TP.
- Precision = TP findings / all findings. A finding matching ANY verified reference counts,
  including maintainability ones: it is a valid comment even if we do not chase those.
- Recall(subset) = covered issues in subset / issues in subset.
Cases that failed or did not run are excluded from every denominator and listed separately.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, Field

from deep_review.evaluation.dataset import SUBSETS, EvalCase, GTIssue
from deep_review.findings import Finding
from deep_review.review import ReviewResult

Matcher = Callable[[Finding, Sequence[GTIssue]], list[str]]


class FindingMatch(BaseModel):
    finding_id: str
    file: str
    line: int
    category: str
    claim: str
    matched: list[str] = Field(default_factory=list)
    new: list[str] = Field(default_factory=list)
    reason: str = ""

    @property
    def tp(self) -> bool:
        return bool(self.new)


class CaseScore(BaseModel):
    case_id: str
    language: str
    findings: list[FindingMatch]
    gt_total: dict[str, int]
    gt_covered: dict[str, int]
    covered_ids: list[str]
    stats: dict[str, Any] = Field(default_factory=dict)


def _norm(path: str) -> str:
    return PurePosixPath(path.strip().lstrip("/").removeprefix("./")).as_posix()


def score_case(
    case: EvalCase,
    result: ReviewResult,
    matcher: Callable[[Finding, Sequence[GTIssue]], tuple[list[str], str]],
) -> CaseScore:
    by_file: dict[str, list[GTIssue]] = defaultdict(list)
    for g in case.ground_truth:
        by_file[_norm(g.file)].append(g)
    covered: set[str] = set()
    matches = []
    for f in result.findings:  # already in the engine's rank order
        ids, reason = matcher(f, by_file.get(_norm(f.file), []))
        new = [i for i in ids if i not in covered]
        covered.update(new)
        matches.append(
            FindingMatch(
                finding_id=f.id,
                file=f.file,
                line=f.line,
                category=f.category,
                claim=f.claim,
                matched=ids,
                new=new,
                reason=reason,
            )
        )
    cat = {g.id: g.category for g in case.ground_truth}
    total = {
        s: sum(1 for g in case.ground_truth if g.category in cats) for s, cats in SUBSETS.items()
    }
    got = {s: sum(1 for i in covered if cat[i] in cats) for s, cats in SUBSETS.items()}
    return CaseScore(
        case_id=case.id,
        language=case.language,
        findings=matches,
        gt_total=total,
        gt_covered=got,
        covered_ids=sorted(covered),
        stats=result.stats,
    )


def _prf(tp: int, n_findings: int, covered: int, total: int) -> dict[str, float | int]:
    p = tp / n_findings if n_findings else 0.0
    r = covered / total if total else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f1, 3),
        "tp_findings": tp,
        "findings": n_findings,
        "covered": covered,
        "gt": total,
    }


def aggregate(scores: Sequence[CaseScore]) -> dict[str, Any]:
    def block(items: Sequence[CaseScore]) -> dict[str, Any]:
        tp = sum(m.tp for s in items for m in s.findings)
        n = sum(len(s.findings) for s in items)
        return {
            subset: _prf(
                tp,
                n,
                sum(s.gt_covered[subset] for s in items),
                sum(s.gt_total[subset] for s in items),
            )
            for subset in SUBSETS
        }

    by_lang: dict[str, list[CaseScore]] = defaultdict(list)
    for s in scores:
        by_lang[s.language or "?"].append(s)

    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"findings": 0, "tp": 0})
    for s in scores:
        for m in s.findings:
            by_cat[m.category]["findings"] += 1
            by_cat[m.category]["tp"] += m.tp

    def total(key: str) -> int:
        return sum(int(s.stats.get(key) or 0) for s in scores)

    durations = sorted(float(s.stats.get("duration_s") or 0) for s in scores)
    stops: dict[str, int] = defaultdict(int)
    for s in scores:
        stops[str(s.stats.get("stop_reason"))] += 1
    return {
        "cases": len(scores),
        "overall": block(scores),
        "by_language": {
            k: block(v)["bugs"] | {"cases": len(v)} for k, v in sorted(by_lang.items())
        },
        "finding_precision_by_category": {
            k: {**v, "precision": round(v["tp"] / v["findings"], 3) if v["findings"] else 0.0}
            for k, v in sorted(by_cat.items())
        },
        "cost": {
            "prompt_tokens": total("prompt_tokens"),
            "cached_prompt_tokens": total("cached_prompt_tokens"),
            "completion_tokens": total("completion_tokens"),
            "tool_calls": total("tool_calls"),
            "turns": total("turns"),
            "seconds_median": durations[len(durations) // 2] if durations else 0.0,
            "seconds_max": durations[-1] if durations else 0.0,
        },
        "stop_reasons": dict(stops),
        "findings_per_case": round(sum(len(s.findings) for s in scores) / len(scores), 2)
        if scores
        else 0.0,
    }


def score_run(
    cases: Sequence[EvalCase],
    results: dict[str, ReviewResult],
    matcher: Callable[[Finding, Sequence[GTIssue]], tuple[list[str], str]],
    *,
    parallel: int = 4,
) -> tuple[list[CaseScore], list[str]]:
    present = [c for c in cases if c.id in results]
    missing = [c.id for c in cases if c.id not in results]
    with ThreadPoolExecutor(max_workers=max(parallel, 1)) as pool:
        scores = list(pool.map(lambda c: score_case(c, results[c.id], matcher), present))
    return scores, missing
