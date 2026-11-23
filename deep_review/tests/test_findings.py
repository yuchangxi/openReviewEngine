from __future__ import annotations

import pytest
from pydantic import ValidationError

from deep_review.diff import parse_unified_diff
from deep_review.findings import Finding, finding_id, position_ok

DIFF = (
    "diff --git a/svc.py b/svc.py\n"
    "--- a/svc.py\n"
    "+++ b/svc.py\n"
    "@@ -10,3 +10,4 @@ def handle():\n"
    " a\n"
    "+b\n"
    " c\n"
    " d\n"
)


def _finding(**kw) -> Finding:
    base = {
        "id": "x",
        "file": "svc.py",
        "line": 11,
        "category": "logic",
        "severity": "major",
        "claim": "b is wrong",
        "evidence": [{"file": "caller.py", "line": 3, "note": "passes None"}],
        "failure_scenario": "None input → TypeError",
        "verdict": "CONFIRMED",
        "confidence": 0.9,
    }
    return Finding.model_validate(base | kw)


def test_schema_round_trip() -> None:
    f = _finding(end_line=12, suggestion="guard None")
    assert Finding.model_validate_json(f.model_dump_json()) == f
    assert f.engine == "deep_review"


@pytest.mark.parametrize(
    "bad",
    [
        {"file": "/etc/passwd"},
        {"file": "../outside.py"},
        {"line": 0},
        {"end_line": 5},
        {"confidence": 1.5},
        {"verdict": "REJECTED"},
        {"category": "style"},
        {"evidence": [{"file": "../x", "line": 1, "note": ""}]},
        {"unexpected": 1},
    ],
)
def test_schema_rejects(bad: dict) -> None:
    with pytest.raises(ValidationError):
        _finding(**bad)


def test_finding_id_is_stable() -> None:
    a = finding_id("deep_review", "svc.py", 11, "b is wrong ")
    assert a == finding_id("deep_review", "svc.py", 11, "b is wrong")
    assert a != finding_id("deep_review", "svc.py", 12, "b is wrong")


def test_position_check() -> None:
    diffs = {fd.path: fd for fd in parse_unified_diff(DIFF)}
    assert position_ok(_finding(line=11), diffs)
    assert position_ok(_finding(line=13), diffs)  # context line inside the hunk
    assert not position_ok(_finding(line=50), diffs)
    assert not position_ok(_finding(file="other.py", line=11), diffs)
    assert position_ok(_finding(file="other.py", line=7), diffs, impact_sites=[("other.py", 7)])
