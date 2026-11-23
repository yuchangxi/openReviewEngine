"""Path guard: every tool path goes through here (PLAN §4.3, §4.6).

Rules: relative paths only; after resolving symlinks the target must stay inside the
worktree; `.git` is off limits; binaries and LFS pointers are refused as text; size capped.
Error messages are written for the model, since they come back as tool results.
"""

from __future__ import annotations

from pathlib import Path

from deep_review.errors import BinaryFileError, FileTooLargeError, PathEscapeError, ToolError

_SNIFF_BYTES = 8192
_LFS_PREFIX = b"version https://git-lfs.github.com/spec/"


def looks_binary(data: bytes) -> bool:
    return b"\x00" in data[:_SNIFF_BYTES]


def is_lfs_pointer(data: bytes) -> bool:
    return data.startswith(_LFS_PREFIX)


class PathGuard:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)

    def resolve(self, rel: str | Path) -> Path:
        s = str(rel).strip()
        if not s or "\x00" in s:
            raise PathEscapeError("invalid path")
        p = Path(s)
        if p.is_absolute():
            raise PathEscapeError(
                f"absolute paths are not allowed: {s!r}; use a path relative to the repository root"
            )
        candidate = (self.root / p).resolve(strict=False)
        if not candidate.is_relative_to(self.root):
            raise PathEscapeError(f"path is outside the repository: {s!r}")
        if ".git" in candidate.relative_to(self.root).parts:
            raise PathEscapeError(f"access to .git is not allowed: {s!r}")
        return candidate

    def relpath(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def read_bytes(self, rel: str | Path, max_bytes: int) -> bytes:
        path = self.resolve(rel)
        if not path.exists():
            raise ToolError(f"no such file: {rel!s}")
        if not path.is_file():
            raise ToolError(f"not a regular file: {rel!s}")
        size = path.stat().st_size
        if size > max_bytes:
            raise FileTooLargeError(
                f"{rel!s} is {size} bytes (limit {max_bytes}); read a line range instead"
            )
        return path.read_bytes()

    def read_text(self, rel: str | Path, max_bytes: int) -> str:
        data = self.read_bytes(rel, max_bytes)
        if looks_binary(data):
            raise BinaryFileError(f"{rel!s} is a binary file")
        if is_lfs_pointer(data):
            raise BinaryFileError(f"{rel!s} is a Git LFS pointer; its content is not available")
        return data.decode("utf-8", errors="replace")
