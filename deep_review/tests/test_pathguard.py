from __future__ import annotations

import os
from pathlib import Path

import pytest

from deep_review.errors import BinaryFileError, FileTooLargeError, PathEscapeError, ToolError
from deep_review.tools import PathGuard


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "src").mkdir(parents=True)
    (r / "src" / "a.py").write_text("x = 1\n")
    (r / ".git").write_text("gitdir: /somewhere\n")
    (r / "bin.dat").write_bytes(b"\x00\x01")
    (r / "lfs.dat").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:00\n")
    (tmp_path / "secret.txt").write_text("secret")
    os.symlink(tmp_path / "secret.txt", r / "escape_link")
    os.symlink("src/a.py", r / "inside_link")
    return r


def test_reads_inside(root: Path) -> None:
    g = PathGuard(root)
    assert g.read_text("src/a.py", 100) == "x = 1\n"
    assert g.read_text("./src/../src/a.py", 100) == "x = 1\n"
    assert g.read_text("inside_link", 100) == "x = 1\n"
    assert g.relpath(root / "src" / "a.py") == "src/a.py"


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../secret.txt",
        "src/../../secret.txt",
        "escape_link",
        ".git",
        "src/../.git/config",
        "",
        "a\x00b",
    ],
)
def test_escapes_are_refused(root: Path, path: str) -> None:
    with pytest.raises(PathEscapeError):
        PathGuard(root).read_text(path, 100)


def test_caps_and_binaries(root: Path) -> None:
    g = PathGuard(root)
    with pytest.raises(FileTooLargeError):
        g.read_text("src/a.py", 3)
    with pytest.raises(BinaryFileError):
        g.read_text("bin.dat", 100)
    with pytest.raises(BinaryFileError, match="LFS"):
        g.read_text("lfs.dat", 100)
    with pytest.raises(ToolError, match="no such file"):
        g.read_text("missing.py", 100)
    with pytest.raises(ToolError, match="not a regular file"):
        g.read_text("src", 100)
