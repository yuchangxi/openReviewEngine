"""Evaluation dataset format (PLAN §6): one JSON object per line, one PR per object.

Ground truth is a list of known issues in the PR head, each anchored to file + line range.
Categories are normalized so datasets from different sources can be scored the same way.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

GTCategory = Literal["defect", "security", "performance", "maintainability", "other"]

# Recall subsets reported by the harness. Our engine deliberately skips maintainability nits.
SUBSETS: dict[str, frozenset[str]] = {
    "all": frozenset({"defect", "security", "performance", "maintainability", "other"}),
    "bugs": frozenset({"defect", "security", "performance"}),
    "defect_security": frozenset({"defect", "security"}),
}


class GTIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    file: str
    line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    text: str
    category: GTCategory
    origin: Literal["human", "ai", "unknown"] = "unknown"
    scope: str = ""
    """Where the evidence lives, e.g. 'Diff Level' / 'File Level' / 'Repo Level' (AACR)."""


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    repo_url: str
    pr_url: str = ""
    base_sha: str
    head_sha: str
    title: str = ""
    description: str = ""
    language: str = ""
    pr_category: str = ""
    changed_lines: int = 0
    ground_truth: list[GTIssue] = Field(default_factory=list)


def load_cases(path: Path | str) -> list[EvalCase]:
    cases = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                cases.append(EvalCase.model_validate_json(line))
            except ValueError as e:
                raise ValueError(f"{path}:{n}: {e}") from e
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate case ids")
    return cases


def save_cases(cases: Iterable[EvalCase], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(c.model_dump_json() + "\n")


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_cases(
    cases: list[EvalCase],
    *,
    languages: Iterable[str] = (),
    ids: Iterable[str] = (),
    limit: int | None = None,
) -> list[EvalCase]:
    langs = {lang.lower() for lang in languages}
    wanted = set(ids)
    out = [
        c
        for c in cases
        if (not langs or c.language.lower() in langs) and (not wanted or c.id in wanted)
    ]
    return out[:limit] if limit else out


def to_json_line(obj: BaseModel) -> str:
    return json.dumps(obj.model_dump(mode="json"), ensure_ascii=False)
