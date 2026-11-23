"""`deep_review eval ...` subcommands.

import-aacr  convert AACR-Bench JSON into our dataset JSONL
run          review every case (resumable) into <runs_dir>/<run_id>/
score        judge findings against ground truth → score.json, report.md, judge_sample.md
compare      before/after table for two scored runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from deep_review.config import Config
from deep_review.evaluation.aacr import import_aacr
from deep_review.evaluation.dataset import file_sha256, load_cases, save_cases, select_cases
from deep_review.evaluation.judge import Judge, line_window_matches
from deep_review.evaluation.metrics import aggregate, score_run
from deep_review.evaluation.report import (
    compare_markdown,
    judge_sample_markdown,
    summary_markdown,
)
from deep_review.evaluation.runner import CaseOutcome, load_results, run_dataset
from deep_review.llm import LLM, OpenAICompatClient, ReplayLLM, ScriptedLLM
from deep_review.review import ReviewResult

EVAL_MAX_COMMENTS = 50
_FAKES = (ScriptedLLM, ReplayLLM)
THRESHOLDS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _above(result: ReviewResult, threshold: float) -> ReviewResult:
    return result.model_copy(
        update={"findings": [f for f in result.findings if f.confidence >= threshold]}
    )


def add_parsers(sub: argparse._SubParsersAction, parents: list[argparse.ArgumentParser]) -> None:
    ev = sub.add_parser("eval", help="evaluation harness").add_subparsers(
        dest="eval_command", required=True
    )

    imp = ev.add_parser(
        "import-aacr", parents=parents, help="convert AACR-Bench positive_samples.json"
    )
    imp.add_argument("src", type=Path)
    imp.add_argument("--out", type=Path, required=True)
    imp.add_argument("--version", default="v2")
    imp.add_argument("--languages", default="", help="comma-separated, e.g. Java,TypeScript")
    imp.add_argument("--no-meta", action="store_true", help="skip fetching PR title/description")

    run = ev.add_parser("run", parents=parents, help="review every case of a dataset")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--languages", default="")
    run.add_argument("--cases", default="", help="comma-separated case ids")
    run.add_argument("--limit", type=int)
    run.add_argument("--parallel", type=int)
    run.add_argument("--no-record", action="store_true")

    sc = ev.add_parser("score", parents=parents, help="judge a finished run")
    sc.add_argument("--dataset", type=Path, required=True)
    sc.add_argument("--run-id", required=True)
    sc.add_argument("--parallel", type=int)

    cmp_ = ev.add_parser("compare", parents=parents, help="compare two scored runs")
    cmp_.add_argument("before")
    cmp_.add_argument("after")


def _split(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _progress(o: CaseOutcome, done: int, total: int) -> None:
    extra = f" — {o.error}" if o.error else ""
    print(f"[{done}/{total}] {o.status:7} {o.case_id} ({o.seconds:.0f}s){extra}", file=sys.stderr)


def run_score(
    cfg: Config,
    dataset: Path,
    run_dir: Path,
    judge_llm: LLM,
    judge_model: str,
    parallel: int,
) -> dict[str, Any]:
    cases = load_cases(dataset)
    manifest = json.loads((run_dir / "run.json").read_text())
    wanted = set(manifest["case_ids"])
    cases = [c for c in cases if c.id in wanted]
    results = load_results(run_dir)

    judge = Judge(judge_llm, run_dir / "judge_cache.jsonl", model_name=judge_model)
    scores, missing = score_run(cases, results, judge.match, parallel=parallel)
    curve = {}
    for t in THRESHOLDS:  # every verdict is cached by now, so this costs no model calls
        kept = {k: _above(r, t) for k, r in results.items()}
        t_scores, _ = score_run(cases, kept, judge.match, parallel=parallel)
        agg = aggregate(t_scores)
        curve[f"{t:.1f}"] = agg["overall"]["bugs"] | {"findings_per_case": agg["findings_per_case"]}
    # Graph runs also keep the candidates as they were before verification: scoring both from
    # the same run isolates what the verifier adds or removes.
    stages: dict[str, Any] = {}
    if any(r.pre_verification for r in results.values()):
        pre = {k: r.model_copy(update={"findings": r.pre_verification}) for k, r in results.items()}
        stages["before_verifier"] = aggregate(
            score_run(cases, pre, judge.match, parallel=parallel)[0]
        )
        stages["after_verifier"] = aggregate(scores)
        conf = {
            k: r.model_copy(
                update={"findings": [f for f in r.findings if f.verdict == "CONFIRMED"]}
            )
            for k, r in results.items()
        }
        stages["confirmed_only"] = aggregate(
            score_run(cases, conf, judge.match, parallel=parallel)[0]
        )
    window = cfg.eval.line_window
    lw_scores, _ = score_run(
        cases, results, lambda f, refs: (line_window_matches(f, refs, window), ""), parallel=1
    )
    summary = {
        "run_id": run_dir.name,
        "dataset": str(dataset),
        "dataset_sha256": file_sha256(dataset),
        "judge_model": judge_model,
        "judge_calls": judge.calls,
        "judge_cache_hits": judge.cache_hits,
        "judge_errors": judge.errors,
        "line_window_size": window,
        "missing": missing,
        "judge": aggregate(scores),
        "line_window": aggregate(lw_scores),
        "threshold_curve": curve,
        "stages": {
            k: v["overall"]["bugs"] | {"findings_per_case": v["findings_per_case"]}
            for k, v in stages.items()
        },
    }
    (run_dir / "score.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "case_scores.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in scores))
    (run_dir / "report.md").write_text(summary_markdown(summary, f"Eval run {run_dir.name}"))
    (run_dir / "judge_sample.md").write_text(
        judge_sample_markdown(scores, {c.id: c for c in cases})
    )
    return summary


def main(args: argparse.Namespace, cfg: Config, llm: LLM | None, judge_llm: LLM | None) -> int:
    cmd = args.eval_command
    if cmd == "import-aacr":
        records = json.loads(args.src.read_text(encoding="utf-8"))
        cases = import_aacr(
            records,
            version=args.version,
            languages=set(_split(args.languages)) or None,
            with_meta=not args.no_meta,
            log=lambda m: print(m, file=sys.stderr),
        )
        save_cases(cases, args.out)
        n_gt = sum(len(c.ground_truth) for c in cases)
        print(f"wrote {len(cases)} cases, {n_gt} ground-truth issues → {args.out}", file=sys.stderr)
        return 0

    runs = cfg.eval.runs_dir
    if cmd == "run":
        cases = select_cases(
            load_cases(args.dataset),
            languages=_split(args.languages),
            ids=_split(args.cases),
            limit=args.limit,
        )
        llm = llm or OpenAICompatClient(cfg.llm)
        # Keep every finding: thresholds are swept at scoring time, from the same run.
        cfg.aggregate.plausible_threshold = 0.0
        cfg.aggregate.max_comments_per_pr = EVAL_MAX_COMMENTS
        verifier_model = cfg.llm.effective_verifier_model
        verifier_llm = (
            OpenAICompatClient(cfg.llm, model=verifier_model)
            if verifier_model != cfg.llm.finder_model and not isinstance(llm, _FAKES)
            else None
        )
        outcomes = run_dataset(
            cases,
            args.dataset,
            runs / args.run_id,
            cfg,
            llm,
            verifier_llm=verifier_llm,
            parallel=args.parallel or cfg.eval.parallel_cases,
            record=not args.no_record,
            progress=_progress,
        )
        failed = [o for o in outcomes if o.status == "failed"]
        print(
            f"{len(outcomes) - len(failed)}/{len(outcomes)} cases ok → {runs / args.run_id}",
            file=sys.stderr,
        )
        return 1 if failed and len(failed) == len(outcomes) else 0

    if cmd == "score":
        judge_model = cfg.eval.judge_model or cfg.llm.effective_verifier_model
        judge_llm = judge_llm or OpenAICompatClient(cfg.llm, model=judge_model)
        summary = run_score(
            cfg,
            args.dataset,
            runs / args.run_id,
            judge_llm,
            judge_model,
            args.parallel or cfg.eval.parallel_cases,
        )
        print((runs / args.run_id / "report.md").read_text())
        return 0 if summary["judge"]["cases"] else 1

    if cmd == "compare":
        a = json.loads((runs / args.before / "score.json").read_text())
        b = json.loads((runs / args.after / "score.json").read_text())
        print(compare_markdown(a, b, (args.before, args.after)))
        return 0
    return 2
