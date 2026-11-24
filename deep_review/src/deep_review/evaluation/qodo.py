"""Importer for Qodo Code Review Benchmark 1.0 (MIT): huggingface.co/datasets/Qodo/PR-Review-Bench.

100 merged upstream PRs replayed as PRs in the github.com/agentic-review-benchmarks forks,
with 580 deliberately injected issues (human double-validated; naturally occurring issues found
during validation were added). Ground truth is therefore much more complete than review-comment
datasets, so measured precision means something.

Each issue is either a violation of a repository rule (`rule_name` set; the rules live in the
fork's root AGENTS.md at the base commit) or a functional/logic bug (`rule_name` empty).
Rule issues map to category "maintainability" so the `bugs` subset scores functional bugs only;
the `all` subset includes rule compliance.

The dataset gives only PR URLs; base/head SHAs come from the public GitHub API
(`fetch_pr_refs`: one list call per fork).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections.abc import Iterable
from typing import Any

from deep_review.evaluation.aacr import MAX_DESCRIPTION_CHARS, parse_pr_url
from deep_review.evaluation.dataset import EvalCase, GTIssue

SOURCE = "qodo-pr-review-bench/1.0"
LANGUAGES = {
    "cal.com": "TypeScript",
    "Ghost": "JavaScript",
    "dify": "Python",
    "firefox-ios": "Swift",
    "prefect": "Python",
    "tauri": "Rust",
    "aspnetcore": "C#",
    "redis": "C",
}
SNIPPET_CHARS = 400
_TITLE_PREFIX = re.compile(r"^Code Review Bench PR #\d+\s*-\s*")


def fetch_pr_refs(owner: str, repos: Iterable[str]) -> dict[str, dict[str, Any]]:
    """PR url → {base_sha, head_sha, title, body} for every PR of the given public repos."""
    token = os.environ.get("GITHUB_TOKEN") or None
    out: dict[str, dict[str, Any]] = {}
    for repo in repos:
        page = 1
        while True:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{owner}/{repo}/pulls"
                f"?state=all&per_page=100&page={page}",
                headers={"Accept": "application/vnd.github+json", "User-Agent": "deep-review-eval"},
            )
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
            for p in data:
                out[p["html_url"]] = {
                    "base_sha": p["base"]["sha"],
                    "head_sha": p["head"]["sha"],
                    "title": p.get("title") or "",
                    "body": p.get("body") or "",
                }
            if len(data) < 100:
                break
            page += 1
    return out


def _int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def convert_record(rec: dict[str, Any], refs: dict[str, Any]) -> EvalCase:
    pr_url = rec["pr_url_to_review"]
    owner, repo, number = parse_pr_url(pr_url)
    case_id = f"qodo__{repo}-{number}"
    gts = []
    for i, it in enumerate(rec.get("issues") or []):
        path = str(it.get("file_path") or "").strip().lstrip("/")
        if path in ("", "None"):  # ~6% of issues give no location; they match in any file
            path, start, end = "", 1, 1
        else:
            start = max(_int(it.get("start_line"), 1), 1)
            end = max(_int(it.get("end_line"), start), start)
        rule = (it.get("rule_name") or "").strip()
        text = f"{(it.get('title') or '').strip()}: {(it.get('description') or '').strip()}"
        snippet = (it.get("problematic_code_snippet") or "").strip()
        if snippet:
            text += f"\nCode: {snippet[:SNIPPET_CHARS]}"
        gts.append(
            GTIssue(
                id=f"{case_id}#{i}",
                file=path,
                line=start,
                end_line=end,
                text=text,
                category="maintainability" if rule else "defect",
                origin="human",
                scope=f"rule: {rule}" if rule else "injected bug",
            )
        )
    body = refs.get("body", "").strip()
    if len(body) > MAX_DESCRIPTION_CHARS:
        body = body[:MAX_DESCRIPTION_CHARS] + "\n…[truncated]"
    return EvalCase(
        id=case_id,
        source=SOURCE,
        repo_url=f"https://github.com/{owner}/{repo}.git",
        pr_url=pr_url,
        base_sha=refs["base_sha"],
        head_sha=refs["head_sha"],
        title=_TITLE_PREFIX.sub("", refs.get("title", "")),
        description=body,
        language=LANGUAGES.get(repo, ""),
        ground_truth=gts,
    )


def import_qodo(
    records: list[dict[str, Any]], refs: dict[str, dict[str, Any]] | None = None
) -> list[EvalCase]:
    if refs is None:
        owners_repos = {parse_pr_url(r["pr_url_to_review"])[:2] for r in records}
        refs = {}
        for owner in {o for o, _ in owners_repos}:
            refs |= fetch_pr_refs(owner, sorted(r for o, r in owners_repos if o == owner))
    missing = [r["pr_url_to_review"] for r in records if r["pr_url_to_review"] not in refs]
    if missing:
        raise ValueError(f"no base/head refs for {len(missing)} PRs, e.g. {missing[0]}")
    return [convert_record(r, refs[r["pr_url_to_review"]]) for r in records]
