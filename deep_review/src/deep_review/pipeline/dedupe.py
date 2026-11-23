"""Merge candidates that several specialists reported for the same problem (PLAN §4 dedupe).

Two candidates in the same file merge when
- their line ranges overlap and their claims share some wording (>= `same_line_similarity`;
  specialists often file one bug under different categories), or
- their ranges are within `window` lines AND they share a category or clearly similar wording
  (>= `min_similarity`). The stricter test keeps different bugs on neighbouring lines apart.
The highest-confidence candidate survives; the specialists that reported it are kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from deep_review.findings import Finding

_WORD = re.compile(r"[a-z0-9_]{3,}")
_STOP = frozenset(
    [
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "when",
        "from",
        "into",
        "will",
        "can",
        "not",
        "are",
        "was",
        "has",
        "have",
        "its",
        "which",
        "than",
        "then",
        "there",
        "their",
        "does",
        "should",
        "could",
        "would",
        "because",
        "only",
        "also",
    ]
)


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def claim_similarity(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


@dataclass
class Group:
    finding: Finding
    sources: list[str] = field(default_factory=list)
    merged: list[Finding] = field(default_factory=list)


def _overlap(a: Finding, b: Finding) -> bool:
    return _near(a, b, 0)


def _near(a: Finding, b: Finding, window: int) -> bool:
    a_end, b_end = a.end_line or a.line, b.end_line or b.line
    return a.file == b.file and a.line <= b_end + window and b.line <= a_end + window


def _same_problem(a: Finding, b: Finding, window: int, min_sim: float, line_sim: float) -> bool:
    if a.file != b.file:
        return False
    sim = claim_similarity(a.claim, b.claim)
    if _overlap(a, b) and sim >= line_sim:
        return True
    return _near(a, b, window) and (a.category == b.category or sim >= min_sim)


def dedupe(
    items: list[tuple[Finding, str]],
    *,
    window: int = 3,
    min_similarity: float = 0.25,
    same_line_similarity: float = 0.15,
) -> list[Group]:
    groups: list[Group] = []
    for f, source in sorted(items, key=lambda x: -x[0].confidence):
        for g in groups:
            if _same_problem(g.finding, f, window, min_similarity, same_line_similarity):
                g.merged.append(f)
                if source not in g.sources:
                    g.sources.append(source)
                break
        else:
            groups.append(Group(f, [source]))
    return groups
