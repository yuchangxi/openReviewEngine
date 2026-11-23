"""Run the engine over an evaluation dataset.

Run directory layout (resumable: finished cases are skipped on re-run):
    <runs>/<run_id>/run.json              manifest: dataset hash, config (no secrets), version
    <runs>/<run_id>/cases/<case>.json     ReviewResult per case
    <runs>/<run_id>/cases/<case>.error    failure message, if the case failed
    <runs>/<run_id>/traces/<case>.jsonl   JSONL trace per case
    <runs>/<run_id>/recordings/<case>.jsonl  model exchanges, replayable offline
"""

from __future__ import annotations

import json
import subprocess
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from deep_review import __version__
from deep_review.config import Config
from deep_review.evaluation.dataset import EvalCase, file_sha256
from deep_review.llm import LLM, RecordingLLM
from deep_review.review import PRInfo, ReviewResult, review_workspace
from deep_review.trace import Tracer
from deep_review.workspace import WorkspaceManager


@dataclass
class CaseOutcome:
    case_id: str
    status: str  # "done" | "skipped" | "failed"
    seconds: float = 0.0
    error: str = ""


def _git_describe() -> str:
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_manifest(run_dir: Path, dataset: Path, cfg: Config, cases: list[EvalCase]) -> None:
    manifest_path = run_dir / "run.json"
    cfg_dump = cfg.model_dump(mode="json")
    manifest = {
        "engine_version": __version__,
        "engine_git": _git_describe(),
        "dataset": str(dataset),
        "dataset_sha256": file_sha256(dataset),
        "case_ids": [c.id for c in cases],
        "config": cfg_dump,  # holds the *name* of the key env var, never the key
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("dataset_sha256") != manifest["dataset_sha256"]:
            raise ValueError(f"{run_dir} was started on a different dataset; use a new run id")
        manifest["started_at"] = old.get("started_at", manifest["started_at"])
        # A partial re-run (--cases) adds to the run; it must not shrink what gets scored.
        manifest["case_ids"] = list(
            dict.fromkeys([*old.get("case_ids", []), *manifest["case_ids"]])
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def run_case(
    case: EvalCase,
    run_dir: Path,
    cfg: Config,
    llm: LLM,
    mgr: WorkspaceManager,
    *,
    record: bool = True,
    verifier_llm: LLM | None = None,
) -> CaseOutcome:
    out_path = run_dir / "cases" / f"{case.id}.json"
    err_path = out_path.with_suffix(".error")
    if out_path.exists():
        return CaseOutcome(case.id, "skipped")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    case_llm: LLM = (
        RecordingLLM(llm, run_dir / "recordings" / f"{case.id}.jsonl") if record else llm
    )
    case_verifier: LLM | None = None
    if verifier_llm is not None:
        case_verifier = (
            RecordingLLM(verifier_llm, run_dir / "recordings" / f"{case.id}.verifier.jsonl")
            if record
            else verifier_llm
        )
    tracer = Tracer(run_dir / "traces" / f"{case.id}.jsonl", run_id=case.id)
    try:
        with (
            tracer,
            tracer.span("eval_case", case=case.id, pr=case.pr_url),
            mgr.checkout(case.repo_url, case.base_sha, case.head_sha) as ws,
        ):
            result: ReviewResult = review_workspace(
                ws,
                case_llm,
                cfg,
                PRInfo(case.title, case.description),
                tracer,
                verifier_llm=case_verifier,
            )
    except Exception as e:  # one bad case must not stop the run
        err_path.write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
        return CaseOutcome(case.id, "failed", time.perf_counter() - t0, f"{type(e).__name__}: {e}")
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp.replace(out_path)  # atomic: a crash never leaves a half-written result
    err_path.unlink(missing_ok=True)
    return CaseOutcome(case.id, "done", time.perf_counter() - t0)


def run_dataset(
    cases: list[EvalCase],
    dataset: Path,
    run_dir: Path,
    cfg: Config,
    llm: LLM,
    *,
    parallel: int = 4,
    record: bool = True,
    verifier_llm: LLM | None = None,
    progress: Callable[[CaseOutcome, int, int], None] = lambda *_: None,
) -> list[CaseOutcome]:
    write_manifest(run_dir, dataset, cfg, cases)
    mgr = WorkspaceManager.from_config(cfg.workspace)
    outcomes: list[CaseOutcome] = []
    with ThreadPoolExecutor(max_workers=max(parallel, 1)) as pool:
        futures = [
            pool.submit(
                run_case, c, run_dir, cfg, llm, mgr, record=record, verifier_llm=verifier_llm
            )
            for c in cases
        ]
        for fut in as_completed(futures):
            outcome = fut.result()
            outcomes.append(outcome)
            progress(outcome, len(outcomes), len(cases))
    return outcomes


def load_results(run_dir: Path) -> dict[str, ReviewResult]:
    results = {}
    for p in sorted((run_dir / "cases").glob("*.json")):
        results[p.stem] = ReviewResult.model_validate_json(p.read_text(encoding="utf-8"))
    return results
