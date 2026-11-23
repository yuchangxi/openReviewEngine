"""Importer for AACR-Bench (Alibaba, Apache-2.0): github.com/alibaba/aacr-bench.

Record mapping (verified against the benchmark's own converter):
    source_commit → base_sha, target_commit → head_sha, githubPrUrl → repo/pr,
    comments[] (side "right" = head lines) → ground truth.
The dataset has no PR title/description; `fetch_pr_meta` can add them from the GitHub API.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from deep_review.evaluation.dataset import EvalCase, GTCategory, GTIssue

SOURCE = "aacr-bench"
_PR_URL = re.compile(r"^https://github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)/?$")
CATEGORY_MAP: dict[str, GTCategory] = {
    "Code Defect": "defect",
    "Security Vulnerability": "security",
    "Performance": "performance",
    "Maintainability and Readability": "maintainability",
}
MAX_DESCRIPTION_CHARS = 4_000


def parse_pr_url(url: str) -> tuple[str, str, int]:
    m = _PR_URL.match(url.strip())
    if not m:
        raise ValueError(f"not a GitHub PR url: {url!r}")
    return m[1], m[2], int(m[3])


def convert_record(rec: dict[str, Any], version: str = "v2") -> EvalCase:
    owner, repo, number = parse_pr_url(rec["githubPrUrl"])
    case_id = f"{owner}__{repo}-{number}"
    gts: list[GTIssue] = []
    for i, c in enumerate(rec.get("comments") or []):
        if c.get("side", "right") != "right" or not (c.get("note") or "").strip():
            continue
        start = int(c.get("from_line") or c.get("to_line") or 1)
        end = int(c.get("to_line") or start)
        start, end = max(min(start, end), 1), max(start, end, 1)
        gts.append(
            GTIssue(
                id=f"{case_id}#{i}",
                file=c["path"].lstrip("/"),
                line=start,
                end_line=end,
                text=c["note"].strip(),
                category=CATEGORY_MAP.get(c.get("category", ""), "other"),
                origin="ai" if c.get("is_ai_comment") else "human",
                scope=c.get("context", ""),
            )
        )
    return EvalCase(
        id=case_id,
        source=f"{SOURCE}/{version}",
        repo_url=f"https://github.com/{owner}/{repo}.git",
        pr_url=rec["githubPrUrl"],
        base_sha=rec["source_commit"],
        head_sha=rec["target_commit"],
        language=rec.get("project_main_language", ""),
        pr_category=rec.get("category", ""),
        changed_lines=int(rec.get("change_line_count") or 0),
        ground_truth=gts,
    )


def fetch_pr_meta(pr_url: str, token: str | None = None) -> tuple[str, str]:
    """(title, description) from the public GitHub API. Sends no user data."""
    owner, repo, number = parse_pr_url(pr_url)
    req = urllib.request.Request(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "deep-review-eval"},
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    body = (data.get("body") or "").strip()
    if len(body) > MAX_DESCRIPTION_CHARS:
        body = body[:MAX_DESCRIPTION_CHARS] + "\n…[truncated]"
    return data.get("title") or "", body


def import_aacr(
    records: list[dict[str, Any]],
    *,
    version: str = "v2",
    languages: set[str] | None = None,
    with_meta: bool = True,
    fetch: Callable[[str, str | None], tuple[str, str]] = fetch_pr_meta,
    log: Callable[[str], None] = lambda _msg: None,
) -> list[EvalCase]:
    token = os.environ.get("GITHUB_TOKEN") or None
    cases = []
    for rec in records:
        if languages and rec.get("project_main_language") not in languages:
            continue
        case = convert_record(rec, version)
        if with_meta:
            try:
                case.title, case.description = fetch(case.pr_url, token)
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                log(f"no title/description for {case.pr_url}: {e}")
        cases.append(case)
    return cases
