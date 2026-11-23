"""Unified diff parser for `git diff` output (also accepts plain `---`/`+++` diffs).

Line numbers are 1-based. Old-side numbers refer to the merge-base version,
new-side numbers to the head version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

LineKind = Literal["add", "del", "ctx"]
FileStatus = Literal["added", "deleted", "modified", "renamed", "copied"]

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")
_C_ESCAPES = {
    "a": 0x07,
    "b": 0x08,
    "t": 0x09,
    "n": 0x0A,
    "v": 0x0B,
    "f": 0x0C,
    "r": 0x0D,
    '"': 0x22,
    "\\": 0x5C,
}


@dataclass(slots=True)
class DiffLine:
    kind: LineKind
    old_no: int | None
    new_no: int | None
    text: str
    no_newline_at_eof: bool = False


@dataclass(slots=True)
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    section: str = ""
    lines: list[DiffLine] = field(default_factory=list)

    def new_range(self) -> tuple[int, int] | None:
        """Inclusive new-side line range, or None for a pure deletion."""
        if self.new_len == 0:
            return None
        return self.new_start, self.new_start + self.new_len - 1


@dataclass(slots=True)
class FileDiff:
    old_path: str | None
    new_path: str | None
    status: FileStatus = "modified"
    is_binary: bool = False
    old_mode: str | None = None
    new_mode: str | None = None
    similarity: int | None = None
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def path(self) -> str:
        """The path to review: new path, or old path for deleted files."""
        p = self.new_path or self.old_path
        assert p is not None
        return p

    @property
    def additions(self) -> int:
        return sum(1 for h in self.hunks for ln in h.lines if ln.kind == "add")

    @property
    def deletions(self) -> int:
        return sum(1 for h in self.hunks for ln in h.lines if ln.kind == "del")

    def added_lines(self) -> list[int]:
        return [ln.new_no for h in self.hunks for ln in h.lines if ln.kind == "add" and ln.new_no]

    def deleted_lines(self) -> list[int]:
        return [ln.old_no for h in self.hunks for ln in h.lines if ln.kind == "del" and ln.old_no]

    def new_ranges(self) -> list[tuple[int, int]]:
        return [r for h in self.hunks if (r := h.new_range()) is not None]

    def in_new_hunk(self, line: int) -> bool:
        return any(start <= line <= end for start, end in self.new_ranges())


def _unquote_c_style(s: str) -> str:
    """Undo git's C-style path quoting: "a/f\\303\\251 \\"x\\"" → a/fé "x"."""
    assert s.startswith('"') and s.endswith('"')
    body, out, i = s[1:-1], bytearray(), 0
    while i < len(body):
        ch = body[i]
        if ch != "\\":
            out += ch.encode("utf-8")
            i += 1
            continue
        nxt = body[i + 1] if i + 1 < len(body) else ""
        if nxt in _C_ESCAPES:
            out.append(_C_ESCAPES[nxt])
            i += 2
        elif re.fullmatch(r"[0-7]{3}", body[i + 1 : i + 4]):
            out.append(int(body[i + 1 : i + 4], 8))
            i += 4
        else:
            out += b"\\"
            i += 1
    return out.decode("utf-8", errors="replace")


def _split_quoted(s: str) -> tuple[str, str]:
    """Split a leading C-quoted token from the rest of the string."""
    i = 1
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == '"':
            return _unquote_c_style(s[: i + 1]), s[i + 1 :].lstrip(" ")
        i += 1
    return s, ""


def _strip_prefix(path: str, prefix: str) -> str:
    return path[len(prefix) :] if path.startswith(prefix) else path


def _parse_marker_path(raw: str, prefix: str) -> str | None:
    """Path from a `--- a/x` / `+++ b/x` line (without the marker)."""
    if raw.startswith('"'):
        path, _ = _split_quoted(raw)
    else:
        path = raw.split("\t", 1)[0]  # plain diffs may append a timestamp after a tab
    if path == "/dev/null":
        return None
    return _strip_prefix(path, prefix)


def _parse_git_header(rest: str) -> tuple[str | None, str | None]:
    """Paths from `diff --git a/x b/y`. Best effort; ---/+++ and rename lines override."""
    if rest.startswith('"'):
        old, remainder = _split_quoted(rest)
        new = _split_quoted(remainder)[0] if remainder.startswith('"') else remainder
        return _strip_prefix(old, "a/"), _strip_prefix(new, "b/")
    # Unquoted, same path on both sides (the common case): "a/<p> b/<p>".
    if rest.startswith("a/") and (len(rest) - 5) % 2 == 0:
        n = (len(rest) - 5) // 2
        p = rest[2 : 2 + n]
        if rest == f"a/{p} b/{p}":
            return p, p
    idx = rest.rfind(" b/")
    if idx != -1:
        return _strip_prefix(rest[:idx], "a/"), rest[idx + 3 :]
    return None, None


def parse_unified_diff(text: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    cur: FileDiff | None = None
    hunk: Hunk | None = None
    old_no = new_no = old_left = new_left = 0
    in_binary_patch = False

    # Split on "\n" only: str.splitlines() would also split on \f, \v, \x1c,   inside code.
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    for raw in lines:
        # Hunk body.
        if hunk is not None and (old_left > 0 or new_left > 0):
            tag, body = raw[:1], raw[1:]
            if tag == "+" and new_left > 0:
                hunk.lines.append(DiffLine("add", None, new_no, body))
                new_no += 1
                new_left -= 1
                continue
            if tag == "-" and old_left > 0:
                hunk.lines.append(DiffLine("del", old_no, None, body))
                old_no += 1
                old_left -= 1
                continue
            if (tag == " " or raw == "") and old_left > 0 and new_left > 0:
                hunk.lines.append(DiffLine("ctx", old_no, new_no, body))
                old_no += 1
                new_no += 1
                old_left -= 1
                new_left -= 1
                continue
            if tag == "\\":
                if hunk.lines:
                    hunk.lines[-1].no_newline_at_eof = True
                continue
            # Malformed/truncated hunk: stop the hunk and treat the line as a header.
            hunk = None
            old_left = new_left = 0

        if raw.startswith("\\"):  # "\ No newline at end of file" right after a hunk's last line
            if hunk is not None and hunk.lines:
                hunk.lines[-1].no_newline_at_eof = True
            continue

        if raw.startswith("diff --git "):
            old, new = _parse_git_header(raw[len("diff --git ") :])
            cur = FileDiff(old_path=old, new_path=new)
            files.append(cur)
            hunk = None
            in_binary_patch = False
            continue

        if in_binary_patch:
            continue

        # Plain diff (no "diff --git" header): "---" opens a new file.
        if raw.startswith("--- ") and (cur is None or cur.hunks):
            cur = FileDiff(old_path=None, new_path=None)
            files.append(cur)
            hunk = None

        if cur is None:
            continue  # preamble (e.g. commit message from `git format-patch`)

        if m := _HUNK_RE.match(raw):
            old_start, old_len = int(m[1]), int(m[2]) if m[2] is not None else 1
            new_start, new_len = int(m[3]), int(m[4]) if m[4] is not None else 1
            hunk = Hunk(old_start, old_len, new_start, new_len, m[5])
            cur.hunks.append(hunk)
            old_no, new_no, old_left, new_left = old_start, new_start, old_len, new_len
            continue

        if raw.startswith("--- "):
            cur.old_path = _parse_marker_path(raw[4:], "a/")
        elif raw.startswith("+++ "):
            cur.new_path = _parse_marker_path(raw[4:], "b/")
        elif raw.startswith("new file mode "):
            cur.status, cur.new_mode = "added", raw.split()[-1]
        elif raw.startswith("deleted file mode "):
            cur.status, cur.old_mode = "deleted", raw.split()[-1]
        elif raw.startswith("old mode "):
            cur.old_mode = raw.split()[-1]
        elif raw.startswith("new mode "):
            cur.new_mode = raw.split()[-1]
        elif raw.startswith(("rename from ", "copy from ")):
            value = raw.split(" ", 2)[2]
            cur.old_path = _split_quoted(value)[0] if value.startswith('"') else value
            cur.status = "renamed" if raw.startswith("rename") else "copied"
        elif raw.startswith(("rename to ", "copy to ")):
            value = raw.split(" ", 2)[2]
            cur.new_path = _split_quoted(value)[0] if value.startswith('"') else value
        elif raw.startswith(("similarity index ", "dissimilarity index ")):
            cur.similarity = int(raw.rsplit(" ", 1)[-1].rstrip("%") or 0)
        elif raw.startswith("Binary files ") and raw.endswith(" differ"):
            cur.is_binary = True
        elif raw.startswith("GIT binary patch"):
            cur.is_binary = True
            in_binary_patch = True

    for fd in files:
        if fd.status == "added" or (fd.old_path is None and fd.new_path is not None):
            fd.status, fd.old_path = "added", None
        elif fd.status == "deleted" or (fd.new_path is None and fd.old_path is not None):
            fd.status, fd.new_path = "deleted", None
    return [fd for fd in files if fd.old_path or fd.new_path]
