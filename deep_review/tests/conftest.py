from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

GIT_ID = [
    "-c",
    "user.name=test",
    "-c",
    "user.email=test@example.com",
    "-c",
    "init.defaultBranch=main",
]


def git(cwd: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", *GIT_ID, *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def write(root: Path, rel: str, content: str | bytes) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content, encoding="utf-8")


UTIL_V1 = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"
UTIL_V2 = "def add(a, b, c=0):\n    return a + b + c\n\n\ndef sub(a, b):\n    return a - b\n"


@pytest.fixture
def upstream(tmp_path: Path) -> dict[str, object]:
    """main: base commit, then one more commit. feature: branched from base, changes util.py,
    adds new.py and a binary file. So merge-base(main, feature) is the base commit."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    git(repo, "init", "-q")
    write(repo, "a.py", "print('a')\n")
    write(repo, "lib/util.py", UTIL_V1)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    base_commit = git(repo, "rev-parse", "HEAD")

    git(repo, "checkout", "-q", "-b", "feature")
    write(repo, "lib/util.py", UTIL_V2)
    write(repo, "new.py", "from lib.util import add\n\nprint(add(1, 2))\n")
    write(repo, "img.bin", b"\x89PNG\x00\x01\x02")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "feature")
    feature_commit = git(repo, "rev-parse", "HEAD")

    git(repo, "checkout", "-q", "main")
    write(repo, "a.py", "print('a2')\n")
    git(repo, "commit", "-q", "-am", "main moves on")
    main_commit = git(repo, "rev-parse", "HEAD")
    return {
        "path": repo,
        "url": str(repo),
        "base_commit": base_commit,
        "feature_commit": feature_commit,
        "main_commit": main_commit,
    }


@pytest.fixture
def ws(tmp_path: Path, upstream: dict[str, object]):
    """A checked-out workspace for main..feature of the `upstream` fixture repo."""
    from deep_review.workspace import WorkspaceManager

    mgr = WorkspaceManager(tmp_path / "cache")
    with mgr.checkout(str(upstream["url"]), "main", "feature") as workspace:
        yield workspace
