"""Finding schema shared with tier 1 and the platform (PLAN §5), plus the position check (§4.5)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from deep_review.diff.parser import FileDiff

Category = Literal["logic", "contract", "concurrency", "resource", "security", "rules"]
Severity = Literal["critical", "major", "minor"]
Verdict = Literal["CONFIRMED", "PLAUSIBLE"]

SEVERITY_RANK: dict[str, int] = {"critical": 0, "major": 1, "minor": 2}


def _check_repo_path(value: str) -> str:
    if not value or value.startswith("/") or "\\" in value or "\x00" in value:
        raise ValueError(f"path must be relative to the repository root: {value!r}")
    if ".." in PurePosixPath(value).parts:
        raise ValueError(f"path must not contain '..': {value!r}")
    return value


RepoPath = Annotated[str, AfterValidator(_check_repo_path)]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: RepoPath
    line: int = Field(ge=1)
    note: str


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    engine: str = "deep_review"
    file: RepoPath
    line: int = Field(ge=1)
    end_line: int | None = Field(default=None, ge=1)
    category: Category
    severity: Severity
    claim: str = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)
    failure_scenario: str = Field(min_length=1)
    suggestion: str | None = None
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _end_after_start(self) -> Finding:
        if self.end_line is not None and self.end_line < self.line:
            raise ValueError(f"end_line ({self.end_line}) is before line ({self.line})")
        return self


def finding_id(engine: str, file: str, line: int, claim: str) -> str:
    """Stable id: the same claim at the same place gets the same id across runs."""
    digest = hashlib.sha256(f"{engine}\0{file}\0{line}\0{claim.strip()}".encode()).hexdigest()
    return f"{engine}-{digest[:16]}"


def position_ok(
    finding: Finding,
    diffs: Mapping[str, FileDiff],
    impact_sites: Iterable[tuple[str, int]] = (),
) -> bool:
    """A finding may only be posted on a line inside a diff hunk (new side),
    or on a call site that the impact stage cited."""
    fd = diffs.get(finding.file)
    if fd is not None and fd.in_new_hunk(finding.line):
        return True
    return (finding.file, finding.line) in set(impact_sites)
