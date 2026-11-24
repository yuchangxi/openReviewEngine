from __future__ import annotations

import threading
from pathlib import Path

import pytest
from conftest import UTIL_V1, git, write

from deep_review.errors import GitError, WorkspaceError
from deep_review.workspace import WorkspaceManager
from deep_review.workspace.git import git_text


def _job_refs(git_dir: Path) -> str:
    return git_text(["for-each-ref", "refs/deep_review"], git_dir=git_dir).strip()


def _worktrees(git_dir: Path) -> list[str]:
    out = git_text(["worktree", "list", "--porcelain"], git_dir=git_dir)
    return [ln for ln in out.splitlines() if ln.startswith("worktree ")]


def test_checkout_pins_shas_and_diffs_from_merge_base(tmp_path: Path, upstream: dict) -> None:
    mgr = WorkspaceManager(tmp_path / "cache")
    with mgr.checkout(upstream["url"], "main", "feature", job_id="job1") as ws:
        assert ws.base_sha == upstream["main_commit"]
        assert ws.head_sha == upstream["feature_commit"]
        assert ws.merge_base == upstream["base_commit"]
        assert (ws.root / "new.py").exists()
        assert (ws.root / "a.py").read_text() == "print('a')\n"  # head's version, not main's

        files = {fd.path: fd for fd in ws.file_diffs()}
        # a.py changed only on main after the branch point → not part of this PR.
        assert set(files) == {"lib/util.py", "new.py", "img.bin"}
        assert files["img.bin"].is_binary
        assert files["lib/util.py"].added_lines() == [1, 2]

        assert ws.show(ws.merge_base, "lib/util.py") == UTIL_V1.encode()
        assert ws.show(ws.merge_base, "new.py") is None
        assert ws.guard.read_text("new.py", 10_000).startswith("from lib.util")
        root, git_dir = ws.root, ws.git_dir
        assert _job_refs(git_dir)

    assert not root.exists()
    assert _job_refs(git_dir) == ""
    assert len(_worktrees(git_dir)) == 1  # only the bare repo itself


def test_cleanup_runs_when_the_job_fails(tmp_path: Path, upstream: dict) -> None:
    mgr = WorkspaceManager(tmp_path / "cache")
    with pytest.raises(RuntimeError), mgr.checkout(upstream["url"], "main", "feature") as ws:
        root = ws.root
        raise RuntimeError("specialist crashed")
    assert not root.exists()
    assert _job_refs(mgr.repo_git_dir(upstream["url"])) == ""


def test_fetch_by_sha_and_cache_reuse(tmp_path: Path, upstream: dict) -> None:
    mgr = WorkspaceManager(tmp_path / "cache")
    with mgr.checkout(upstream["url"], "main", "feature"):
        pass
    marker = mgr.repo_git_dir(upstream["url"]) / "deep_review.ready"
    mtime = marker.stat().st_mtime_ns
    with mgr.checkout(upstream["url"], upstream["base_commit"], upstream["feature_commit"]) as ws:
        assert ws.merge_base == upstream["base_commit"]
    assert marker.stat().st_mtime_ns == mtime  # repo was not re-initialised


def test_parallel_jobs_on_one_repo(tmp_path: Path, upstream: dict) -> None:
    mgr = WorkspaceManager(tmp_path / "cache")
    errors: list[BaseException] = []
    seen: list[str] = []

    def job(i: int) -> None:
        try:
            with mgr.checkout(upstream["url"], "main", "feature", job_id=f"p{i}") as ws:
                seen.append(ws.head_sha)
                assert {fd.path for fd in ws.file_diffs()} == {"lib/util.py", "new.py", "img.bin"}
        except BaseException as e:  # surface assertion errors from threads
            errors.append(e)

    threads = [threading.Thread(target=job, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert seen == [upstream["feature_commit"]] * 6
    git_dir = mgr.repo_git_dir(upstream["url"])
    assert _job_refs(git_dir) == "" and len(_worktrees(git_dir)) == 1


def test_missing_ref_and_unrelated_history(tmp_path: Path, upstream: dict) -> None:
    repo: Path = upstream["path"]
    git(repo, "checkout", "-q", "--orphan", "orphan")
    git(repo, "rm", "-rqf", ".")
    write(repo, "other.txt", "unrelated\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "orphan")

    mgr = WorkspaceManager(tmp_path / "cache")
    with pytest.raises(GitError, match="fetch"), mgr.checkout(upstream["url"], "main", "nope"):
        pass
    with (
        pytest.raises(WorkspaceError, match="no common ancestor"),
        mgr.checkout(upstream["url"], "main", "orphan"),
    ):
        pass
    git_dir = mgr.repo_git_dir(upstream["url"])
    assert _job_refs(git_dir) == "" and len(_worktrees(git_dir)) == 1


@pytest.mark.parametrize(
    ("url", "base", "head"),
    [
        ("--upload-pack=touch /tmp/x", "main", "feature"),
        ("ext::sh -c touch% /tmp/x", "main", "feature"),
        ("REPO", "--output=/tmp/x", "feature"),
        ("REPO", "main", "a:b"),
        ("REPO", "main", "refs/heads/../x"),
    ],
)
def test_hostile_inputs_are_rejected(tmp_path: Path, upstream: dict, url, base, head) -> None:
    url = upstream["url"] if url == "REPO" else url
    mgr = WorkspaceManager(tmp_path / "cache")
    with pytest.raises(GitError, match="invalid|not allowed"), mgr.checkout(url, base, head):
        pass
    assert not (tmp_path / "cache").exists()  # rejected before touching disk


def test_sweep_removes_worktrees_of_killed_jobs(tmp_path: Path, upstream: dict) -> None:
    import os

    mgr = WorkspaceManager(tmp_path / "cache")
    cm = mgr.checkout(upstream["url"], "main", "feature", job_id="killed")
    ws = cm.__enter__()  # simulate a process killed mid-review: no __exit__, no cleanup
    git_dir = ws.git_dir
    assert ws.root.exists() and _job_refs(git_dir)
    with mgr.checkout(upstream["url"], "main", "feature", job_id="live") as live:
        assert mgr.sweep_stale(max_age_s=3600) == []  # fresh worktrees are left alone
        old = 1_000_000_000
        os.utime(ws.root, (old, old))
        assert mgr.sweep_stale(max_age_s=3600) == ["killed"]
        assert not ws.root.exists() and live.root.exists()  # the live job is untouched
        assert "killed" not in _job_refs(git_dir) and "live" in _job_refs(git_dir)
    assert _job_refs(git_dir) == "" and len(_worktrees(git_dir)) == 1
    cm.__exit__(None, None, None)  # a late cleanup after the sweep must be harmless
    assert _job_refs(git_dir) == ""


def test_only_transient_git_errors_are_retried(tmp_path, monkeypatch) -> None:
    from deep_review.errors import GitError
    from deep_review.workspace import manager as m

    assert m.is_transient(GitError("error: RPC failed; curl 56 GnuTLS recv error (-9)"))
    assert m.is_transient(GitError("fatal: early EOF"))
    assert not m.is_transient(GitError("fatal: couldn't find remote ref refs/heads/nope"))
    assert not m.is_transient(GitError("upload-pack: not our ref deadbeef"))

    monkeypatch.setattr(m, "RETRY_BACKOFF_S", 0.0)
    mgr = m.WorkspaceManager(tmp_path)
    calls: list[int] = []
    undone: list[int] = []

    def flaky() -> None:
        calls.append(1)
        if len(calls) < 3:
            raise GitError("fatal: the remote end hung up unexpectedly")

    mgr._with_retries("x", flaky, undo=lambda: undone.append(1))
    assert len(calls) == 3 and len(undone) == 2

    def fatal() -> None:
        calls.append(1)
        raise GitError("fatal: couldn't find remote ref refs/heads/nope")

    calls.clear()
    with pytest.raises(GitError):
        mgr._with_retries("x", fatal)
    assert len(calls) == 1
