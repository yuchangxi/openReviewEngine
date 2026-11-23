"""Safe git invocation: fixed env/config, timeouts, no prompts, no hooks, argument validation."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

from deep_review.errors import GitError

GIT_ENV = {
    "GIT_TERMINAL_PROMPT": "0",  # never block on a credential prompt
    "GIT_LFS_SKIP_SMUDGE": "1",  # LFS pointers stay pointers
    "GIT_OPTIONAL_LOCKS": "0",
}
GIT_CONFIG = (
    "core.hooksPath=/dev/null",  # no hooks, ever
    "core.quotePath=false",
    "core.fsmonitor=false",
    "gc.auto=0",  # no surprise gc while other jobs share the repo
    "maintenance.auto=false",
    "advice.detachedHead=false",
    "protocol.ext.allow=never",
    "diff.external=",
)

_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_BAD_REF_CHARS = re.compile(r"[\x00-\x20\x7f~^:?*\[\\]")


def validate_ref(ref: str) -> str:
    """Accept a SHA or a ref name; reject anything that could be read as an option or a refspec."""
    if _SHA_RE.match(ref):
        return ref
    if (
        not ref
        or ref.startswith(("-", "/", "+"))
        or ref.endswith(("/", ".", ".lock"))
        or ".." in ref
        or "@{" in ref
        or "//" in ref
        or _BAD_REF_CHARS.search(ref)
    ):
        raise GitError(f"invalid git ref: {ref!r}")
    return ref


def validate_remote_url(url: str) -> str:
    if not url or url.startswith("-") or "\x00" in url or "\n" in url:
        raise GitError(f"invalid repository url: {url!r}")
    if "::" in url.split("/", 1)[0]:  # transport helpers like ext::
        raise GitError(f"transport helper urls are not allowed: {url!r}")
    return url


def run_git(
    args: Sequence[str],
    *,
    git_dir: Path | None = None,
    cwd: Path | None = None,
    timeout: float = 120.0,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    cmd = ["git"]
    for item in GIT_CONFIG:
        cmd += ["-c", item]
    if git_dir is not None:
        cmd += ["--git-dir", str(git_dir)]
    cmd += list(args)
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env={**os.environ, **GIT_ENV},
            capture_output=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as e:
        raise GitError(f"git {args[0]} timed out after {timeout:.0f}s") from e
    if check and proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise GitError(f"git {' '.join(args[:3])} failed ({proc.returncode}): {stderr[-2000:]}")
    return proc


def git_text(args: Sequence[str], **kwargs) -> str:
    return run_git(args, **kwargs).stdout.decode("utf-8", errors="replace")
