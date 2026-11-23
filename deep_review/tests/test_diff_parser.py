from __future__ import annotations

import os
from pathlib import Path

from conftest import git, write

from deep_review.diff import parse_unified_diff


def _real_diff(tmp_path: Path) -> str:
    """Build a diff with real git covering the awkward cases."""
    repo = tmp_path / "r"
    repo.mkdir()
    git(repo, "init", "-q")
    write(repo, "mod.py", "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\n")
    write(repo, "gone.txt", "bye\n")
    write(repo, "old_name.py", "".join(f"line {i}\n" for i in range(20)))
    write(repo, "noeol.txt", "x\ny")
    write(repo, "script.sh", "echo hi\n")
    write(repo, "ff.py", "x = 1\n\f\ny = 2\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "one")

    write(repo, "mod.py", "a\nB\nc\nd\ne\nf\ng\nh\ni\nj\nk\n")
    (repo / "gone.txt").unlink()
    (repo / "old_name.py").rename(repo / "new name.py")
    text = (repo / "new name.py").read_text().replace("line 10\n", "line ten\n")
    write(repo, "new name.py", text)
    write(repo, "noeol.txt", "x\nz")
    write(repo, "café.py", "print('é')\n")
    write(repo, "blob.bin", b"\x00\x01\x02binary")
    os.chmod(repo / "script.sh", 0o755)
    write(repo, "ff.py", "x = 1\n\f\ny = 3\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "two")
    return (
        git(
            repo,
            "-c",
            "core.quotePath=true",
            "diff",
            "--no-color",
            "-M",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            "HEAD~1",
            "HEAD",
        )
        + "\n"
    )


def test_real_git_diff(tmp_path: Path) -> None:
    files = {fd.path: fd for fd in parse_unified_diff(_real_diff(tmp_path))}
    assert set(files) == {
        "mod.py",
        "gone.txt",
        "new name.py",
        "noeol.txt",
        "café.py",
        "blob.bin",
        "script.sh",
        "ff.py",
    }

    mod = files["mod.py"]
    assert mod.status == "modified"
    assert mod.added_lines() == [2, 11]
    assert mod.deleted_lines() == [2]
    assert mod.in_new_hunk(11) and not mod.in_new_hunk(500)

    assert files["gone.txt"].status == "deleted"
    assert files["gone.txt"].new_path is None
    assert files["gone.txt"].new_ranges() == []

    ren = files["new name.py"]
    assert ren.status == "renamed" and ren.old_path == "old_name.py"
    assert ren.added_lines() == [11]

    noeol = files["noeol.txt"]
    last_del = [ln for h in noeol.hunks for ln in h.lines if ln.kind == "del"][-1]
    last_add = [ln for h in noeol.hunks for ln in h.lines if ln.kind == "add"][-1]
    assert last_del.text == "y" and last_del.no_newline_at_eof
    assert last_add.text == "z" and last_add.no_newline_at_eof

    assert files["café.py"].status == "added"  # path was C-quoted with octal escapes
    assert files["café.py"].old_path is None
    assert files["blob.bin"].is_binary and files["blob.bin"].status == "added"

    sh = files["script.sh"]
    assert sh.old_mode == "100644" and sh.new_mode == "100755" and not sh.hunks

    ff = files["ff.py"]  # a form feed must not split lines
    assert ff.added_lines() == [3]
    assert [ln.text for ln in ff.hunks[0].lines if ln.kind == "ctx"] == ["x = 1", "\f"]


def test_plain_diff_without_git_header() -> None:
    text = (
        "--- a/x.py\t2024-01-01 00:00:00\n"
        "+++ b/x.py\t2024-01-02 00:00:00\n"
        "@@ -1 +1,2 @@\n"
        " keep\n"
        "+added\n"
        "--- a/y.py\n"
        "+++ b/y.py\n"
        "@@ -3,2 +3 @@\n"
        "-gone\n"
        " stay\n"
    )
    files = parse_unified_diff(text)
    assert [f.path for f in files] == ["x.py", "y.py"]
    assert files[0].added_lines() == [2]
    assert files[1].deleted_lines() == [3]
    assert files[1].new_ranges() == [(3, 3)]


def test_removed_line_starting_with_dashes_is_not_a_header() -> None:
    text = (
        "diff --git a/s.sql b/s.sql\n"
        "--- a/s.sql\n"
        "+++ b/s.sql\n"
        "@@ -1,2 +1,2 @@\n"
        "--- a comment\n"
        "+-- a new comment\n"
        " select 1;\n"
    )
    (fd,) = parse_unified_diff(text)
    assert fd.deletions == 1 and fd.additions == 1
    assert fd.hunks[0].lines[0].text == "-- a comment"


def test_preamble_and_truncated_hunk_are_tolerated() -> None:
    text = (
        "From abc Mon Sep 17 00:00:00 2001\n"
        "Subject: [PATCH] x\n"
        "diff --git a/f b/f\n"
        "--- a/f\n"
        "+++ b/f\n"
        "@@ -1,3 +1,3 @@\n"
        " a\n"
        "diff --git a/g b/g\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/g\n"
        "@@ -0,0 +1 @@\n"
        "+g\n"
    )
    files = parse_unified_diff(text)
    assert [f.path for f in files] == ["f", "g"]
    assert files[1].status == "added" and files[1].added_lines() == [1]
