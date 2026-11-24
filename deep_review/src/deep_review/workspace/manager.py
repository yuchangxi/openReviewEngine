"""Service-side checkout (PLAN §4.1).

Layout under `cache_dir`:
    repos/<slug>-<hash>.git   cached bare partial repo per remote url (blobs fetched lazily)
    locks/<slug>-<hash>.lock  flock held while fetching / changing the repo's refs
    worktrees/<job_id>/       detached worktree at the pinned head SHA, removed after the job

Each job fetches base and head into private refs (refs/deep_review/jobs/<job_id>/...),
pins their SHAs, computes the merge-base, and checks out head. The diff the engine reviews
is merge-base..head, i.e. what the PR adds — not what happened on the base branch since.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import re
import shutil
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from deep_review.config import WorkspaceConfig
from deep_review.diff.parser import FileDiff, parse_unified_diff
from deep_review.errors import GitError, WorkspaceError
from deep_review.tools.pathguard import PathGuard
from deep_review.workspace.git import git_text, run_git, validate_ref, validate_remote_url

log = logging.getLogger(__name__)

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_READY_MARKER = "deep_review.ready"
NETWORK_ATTEMPTS = 3
"""Fetches and lazy-blob checkouts go over the network; transient TLS/HTTP failures are
retried with backoff (RETRY_BACKOFF_S * attempt)."""
RETRY_BACKOFF_S = 5.0
_TRANSIENT = re.compile(
    r"RPC failed|TLS|SSL|early EOF|remote end hung up|unexpected disconnect|index-pack|"
    r"Connection (?:reset|refused|timed out)|Could not resolve host|Operation timed out|"
    r"timed out|HTTP/?\S* 5\d\d|returned error: 5\d\d|curl \d+",
    re.IGNORECASE,
)


def is_transient(err: Exception) -> bool:
    return bool(_TRANSIENT.search(str(err)))


@dataclass(frozen=True)
class Workspace:
    repo_url: str
    git_dir: Path
    root: Path
    job_id: str
    base_ref: str
    head_ref: str
    base_sha: str
    head_sha: str
    merge_base: str
    git_timeout_s: float = 120.0

    @cached_property
    def guard(self) -> PathGuard:
        return PathGuard(self.root)

    def diff_text(self, context_lines: int = 3) -> str:
        return git_text(
            [
                "diff",
                "--no-color",
                "--no-ext-diff",
                "--no-textconv",
                "-M",
                "--src-prefix=a/",
                "--dst-prefix=b/",
                f"--unified={context_lines}",
                self.merge_base,
                self.head_sha,
                "--",
            ],
            git_dir=self.git_dir,
            timeout=self.git_timeout_s,
        )

    def file_diffs(self, context_lines: int = 3) -> list[FileDiff]:
        return parse_unified_diff(self.diff_text(context_lines))

    def show(self, rev: str, path: str) -> bytes | None:
        """File content at `rev` (e.g. the merge-base for the old version); None if absent."""
        validate_ref(rev)
        if not path or path.startswith(("-", "/")) or "\x00" in path:
            raise WorkspaceError(f"invalid path: {path!r}")
        proc = run_git(
            ["cat-file", "blob", f"{rev}:{path}"],
            git_dir=self.git_dir,
            timeout=self.git_timeout_s,
            check=False,
        )
        return proc.stdout if proc.returncode == 0 else None


def _repo_key(url: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", url.rstrip("/").removesuffix(".git")).strip("-")
    return f"{slug[-48:]}-{hashlib.sha256(url.encode()).hexdigest()[:12]}"


@contextmanager
def _file_lock(path: Path, timeout_s: float) -> Generator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as fh:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise WorkspaceError(
                        f"timed out after {timeout_s:.0f}s waiting for {path}"
                    ) from None
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


class WorkspaceManager:
    def __init__(
        self,
        cache_dir: Path | str,
        *,
        git_timeout_s: float = 120.0,
        fetch_timeout_s: float = 600.0,
        lock_timeout_s: float = 3_600.0,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.git_timeout_s = git_timeout_s
        self.fetch_timeout_s = fetch_timeout_s
        self.lock_timeout_s = lock_timeout_s

    @classmethod
    def from_config(cls, cfg: WorkspaceConfig) -> WorkspaceManager:
        return cls(
            cfg.cache_dir,
            git_timeout_s=cfg.git_timeout_s,
            fetch_timeout_s=cfg.fetch_timeout_s,
            lock_timeout_s=cfg.lock_timeout_s,
        )

    def repo_git_dir(self, url: str) -> Path:
        return self.cache_dir / "repos" / f"{_repo_key(url)}.git"

    def _lock_path(self, url: str) -> Path:
        return self.cache_dir / "locks" / f"{_repo_key(url)}.lock"

    def _ensure_repo(self, git_dir: Path, url: str) -> None:
        """Create the cached bare partial repo once. Caller holds the repo lock."""
        if (git_dir / _READY_MARKER).exists():
            return
        if git_dir.exists():  # left over from an interrupted init
            shutil.rmtree(git_dir)
        git_dir.parent.mkdir(parents=True, exist_ok=True)
        t = self.git_timeout_s
        run_git(["init", "--bare", "--quiet", str(git_dir)], timeout=t)
        # Partial clone, configured exactly as `git clone --bare --filter=blob:none` would:
        # commits and trees on fetch, blobs on demand from the promisor remote.
        for key, value in (
            ("core.repositoryformatversion", "1"),
            ("extensions.partialClone", "origin"),
            ("remote.origin.url", url),
            ("remote.origin.promisor", "true"),
            ("remote.origin.partialclonefilter", "blob:none"),
        ):
            run_git(["config", key, value], git_dir=git_dir, timeout=t)
        (git_dir / _READY_MARKER).touch()

    def _with_retries(self, what: str, fn, undo=None) -> None:  # noqa: ANN001
        for attempt in range(1, NETWORK_ATTEMPTS + 1):
            try:
                fn()
                return
            except GitError as e:
                if attempt == NETWORK_ATTEMPTS or not is_transient(e):
                    raise
                log.warning(
                    "%s failed (attempt %d/%d), retrying: %s", what, attempt, NETWORK_ATTEMPTS, e
                )
                if undo:
                    undo()
                time.sleep(RETRY_BACKOFF_S * attempt)

    def _fetch(self, git_dir: Path, refspecs: list[str]) -> None:
        self._with_retries("fetch", lambda: self._fetch_once(git_dir, refspecs))

    def _fetch_once(self, git_dir: Path, refspecs: list[str]) -> None:
        run_git(
            [
                "fetch",
                "--quiet",
                "--no-tags",
                "--no-write-fetch-head",
                "--filter=blob:none",
                "--force",
                "origin",
                "--",
                *refspecs,
            ],
            git_dir=git_dir,
            timeout=self.fetch_timeout_s,
        )

    def _prefetch_diff_blobs(self, git_dir: Path, merge_base: str, head_sha: str) -> None:
        proc = run_git(
            ["diff", "--quiet", "--no-ext-diff", "--no-textconv", merge_base, head_sha],
            git_dir=git_dir,
            timeout=self.fetch_timeout_s,
            check=False,
        )
        if proc.returncode not in (0, 1):  # 1 just means "there are differences"
            err = proc.stderr.decode(errors="replace")[-2000:]
            raise GitError(f"git diff (blob prefetch) failed ({proc.returncode}): {err}")

    def _rev(self, git_dir: Path, ref: str) -> str:
        return git_text(
            ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            git_dir=git_dir,
            timeout=self.git_timeout_s,
        ).strip()

    @contextmanager
    def checkout(
        self,
        repo_url: str,
        base_ref: str,
        head_ref: str,
        *,
        job_id: str | None = None,
    ) -> Generator[Workspace]:
        validate_remote_url(repo_url)
        validate_ref(base_ref)
        validate_ref(head_ref)
        job_id = job_id or uuid.uuid4().hex[:12]
        if not _JOB_ID_RE.match(job_id):
            raise WorkspaceError(f"invalid job id: {job_id!r}")

        git_dir = self.repo_git_dir(repo_url)
        worktree = self.cache_dir / "worktrees" / job_id
        job_refs = f"refs/deep_review/jobs/{job_id}"
        if worktree.exists():
            raise WorkspaceError(f"worktree for job {job_id} already exists: {worktree}")

        try:
            # The lock also covers the checkout: its lazy blob fetch is the expensive network
            # step, and serializing it lets later jobs on the same repo reuse those blobs
            # instead of downloading the same files concurrently.
            with _file_lock(self._lock_path(repo_url), self.lock_timeout_s):
                self._ensure_repo(git_dir, repo_url)
                self._fetch(
                    git_dir, [f"+{base_ref}:{job_refs}/base", f"+{head_ref}:{job_refs}/head"]
                )
                base_sha = self._rev(git_dir, f"{job_refs}/base")
                head_sha = self._rev(git_dir, f"{job_refs}/head")
                proc = run_git(
                    ["merge-base", base_sha, head_sha],
                    git_dir=git_dir,
                    timeout=self.git_timeout_s,
                    check=False,
                )
                if proc.returncode != 0:
                    raise WorkspaceError(f"{base_ref} and {head_ref} have no common ancestor")
                merge_base = proc.stdout.decode().strip()

                worktree.parent.mkdir(parents=True, exist_ok=True)
                self._with_retries(
                    "checkout",
                    lambda: run_git(
                        ["worktree", "add", "--detach", "--quiet", str(worktree), head_sha],
                        git_dir=git_dir,
                        timeout=self.fetch_timeout_s,  # checkout may lazily fetch many blobs
                    ),
                    undo=lambda: self._remove_worktree(git_dir, worktree),
                )
                # The diff needs the base side of every changed file, which the checkout did
                # not download; fetch it now (retried) so later diffs work offline.
                self._with_retries(
                    "prefetch diff blobs",
                    lambda: self._prefetch_diff_blobs(git_dir, merge_base, head_sha),
                )
            yield Workspace(
                repo_url=repo_url,
                git_dir=git_dir,
                root=worktree,
                job_id=job_id,
                base_ref=base_ref,
                head_ref=head_ref,
                base_sha=base_sha,
                head_sha=head_sha,
                merge_base=merge_base,
                git_timeout_s=self.git_timeout_s,
            )
        finally:
            self._cleanup(git_dir, worktree, job_refs)

    def sweep_stale(self, max_age_s: float = 6 * 3600) -> list[str]:
        """Remove worktrees (and their job refs) left behind by killed processes.

        A job normally cleans up in `finally`; SIGKILL, OOM or power loss skip that. A worktree
        untouched for `max_age_s` (far longer than any review) is treated as abandoned.
        Returns the job ids removed. Best effort; never raises."""
        removed: list[str] = []
        root = self.cache_dir / "worktrees"
        if not root.is_dir():
            return removed
        now = time.time()
        for wt in root.iterdir():
            try:
                if not wt.is_dir() or now - wt.stat().st_mtime < max_age_s:
                    continue
                git_dir: Path | None = None
                dot_git = wt / ".git"
                if dot_git.is_file():  # "gitdir: <repo>.git/worktrees/<name>"
                    target = Path(dot_git.read_text().split(":", 1)[1].strip())
                    git_dir = target.parent.parent
                if git_dir is not None and git_dir.is_dir():
                    self._cleanup(git_dir, wt, f"refs/deep_review/jobs/{wt.name}")
                shutil.rmtree(wt, ignore_errors=True)
                removed.append(wt.name)
            except OSError as e:
                log.warning("could not sweep %s: %s", wt, e)
        if removed:
            log.info("swept %d stale worktrees: %s", len(removed), removed)
        return removed

    def _remove_worktree(self, git_dir: Path, worktree: Path) -> None:
        t = self.git_timeout_s
        try:
            if worktree.exists():
                run_git(
                    ["worktree", "remove", "--force", str(worktree)], git_dir=git_dir, timeout=t
                )
        except GitError as e:
            log.warning("worktree remove failed, deleting directory instead: %s", e)
            shutil.rmtree(worktree, ignore_errors=True)
        run_git(["worktree", "prune"], git_dir=git_dir, timeout=t, check=False)

    def _cleanup(self, git_dir: Path, worktree: Path, job_refs: str) -> None:
        """Best effort; never raises, so it cannot mask the job's own exception."""
        if not git_dir.exists():
            return
        t = self.git_timeout_s
        self._remove_worktree(git_dir, worktree)
        try:
            refs = git_text(
                ["for-each-ref", "--format=%(refname)", job_refs], git_dir=git_dir, timeout=t
            ).split()
            for ref in refs:
                run_git(["update-ref", "-d", ref], git_dir=git_dir, timeout=t, check=False)
        except GitError as e:
            log.warning("could not delete job refs %s: %s", job_refs, e)
