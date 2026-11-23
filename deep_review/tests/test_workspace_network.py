"""M0 done-condition: a worktree is created and cleaned up for a real public repo PR.

Run with: uv run pytest -m network
Override the target with DEEP_REVIEW_TEST_REPO / _BASE / _HEAD.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from deep_review.diff import select_files
from deep_review.workspace import WorkspaceManager
from deep_review.workspace.git import git_text, run_git

REPO = os.environ.get("DEEP_REVIEW_TEST_REPO", "https://github.com/pallets/itsdangerous.git")
BASE = os.environ.get("DEEP_REVIEW_TEST_BASE", "main")
HEAD = os.environ.get("DEEP_REVIEW_TEST_HEAD", "refs/pull/442/head")
SHA = re.compile(r"^[0-9a-f]{40}$")


@pytest.mark.network
def test_public_pr_checkout(tmp_path: Path) -> None:
    mgr = WorkspaceManager(tmp_path / "cache")
    with mgr.checkout(REPO, BASE, HEAD, job_id="net1") as ws:
        assert SHA.match(ws.base_sha) and SHA.match(ws.head_sha) and SHA.match(ws.merge_base)
        is_ancestor = run_git(
            ["merge-base", "--is-ancestor", ws.merge_base, ws.head_sha],
            git_dir=ws.git_dir,
            check=False,
        )
        assert is_ancestor.returncode == 0

        # Partial clone: blobs are fetched lazily, not up front.
        assert (
            git_text(["config", "extensions.partialclone"], git_dir=ws.git_dir).strip() == "origin"
        )

        diffs = ws.file_diffs()
        assert diffs, "PR has no changes against its merge-base"
        sel = select_files(diffs)
        for fd in sel.selected:
            if fd.status != "deleted":
                assert (ws.root / fd.path).is_file(), fd.path
            if fd.status == "modified":
                assert ws.show(ws.merge_base, fd.path) is not None, fd.path
        root, git_dir = ws.root, ws.git_dir
        print(
            f"\n{REPO} {BASE}..{HEAD}: merge-base {ws.merge_base[:10]}, head {ws.head_sha[:10]}, "
            f"{len(diffs)} files, {len(sel.selected)} selected, "
            f"skipped {[(f.path, r.value) for f, r in sel.skipped]}"
        )

    assert not root.exists()
    assert git_text(["for-each-ref", "refs/deep_review"], git_dir=git_dir).strip() == ""
