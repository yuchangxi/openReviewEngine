"""Command line: `deep_review review --repo <url|path> --base <ref> --head <ref>`.

Prints the ReviewResult JSON to stdout (or --out); a short summary goes to stderr.
Exit codes: 0 ok, 1 review failed, 2 bad configuration or arguments.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from deep_review.config import Config, load_config
from deep_review.errors import ConfigError, DeepReviewError
from deep_review.evaluation import commands as eval_commands
from deep_review.llm import LLM, OpenAICompatClient, RecordingLLM, ReplayLLM
from deep_review.review import PRInfo, review_workspace
from deep_review.trace import Tracer
from deep_review.workspace import WorkspaceManager


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, help="TOML config file")
    common.add_argument("-v", "--verbose", action="store_true")
    p = argparse.ArgumentParser(prog="deep_review")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("review", parents=[common], help="review base..head of a repository")
    r.add_argument("--repo", required=True, help="clone URL or local path")
    r.add_argument("--base", required=True, help="base branch, ref or SHA")
    r.add_argument("--head", required=True, help="head ref (e.g. refs/pull/12/head) or SHA")
    r.add_argument("--title", default="")
    desc = r.add_mutually_exclusive_group()
    desc.add_argument("--description", default="")
    desc.add_argument("--description-file", type=Path)
    r.add_argument("--out", type=Path, help="write result JSON here instead of stdout")
    mock = r.add_mutually_exclusive_group()
    mock.add_argument("--replay", type=Path, help="answer from a recording instead of the model")
    mock.add_argument("--record", type=Path, help="record every model exchange to this JSONL")
    r.add_argument("--max-turns", type=int)
    r.add_argument("--job-id")
    eval_commands.add_parsers(sub, parents=[common])
    return p


def _build_llm(cfg: Config, args: argparse.Namespace) -> LLM:
    if args.replay:
        return ReplayLLM(args.replay)
    llm: LLM = OpenAICompatClient(cfg.llm)
    if args.record:
        llm = RecordingLLM(llm, args.record)
    return llm


def _build_verifier(cfg: Config, args: argparse.Namespace, llm: LLM) -> LLM | None:
    """A separate client only when the verifier model differs and we talk to a real endpoint."""
    model = cfg.llm.effective_verifier_model
    if (
        args.replay
        or model == cfg.llm.finder_model
        or not isinstance(llm, (OpenAICompatClient, RecordingLLM))
    ):
        return None
    verifier: LLM = OpenAICompatClient(cfg.llm, model=model)
    if args.record:
        verifier = RecordingLLM(verifier, args.record.with_suffix(".verifier.jsonl"))
    return verifier


def main(
    argv: list[str] | None = None, *, llm: LLM | None = None, judge_llm: LLM | None = None
) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command == "eval":
        try:
            cfg = load_config(args.config)
            return eval_commands.main(args, cfg, llm, judge_llm)
        except (ConfigError, DeepReviewError, ValueError) as e:
            print(f"eval failed: {e}", file=sys.stderr)
            return 2
    try:
        cfg = load_config(args.config)
        if args.max_turns:
            cfg.budgets.max_turns = args.max_turns
        llm = llm or _build_llm(cfg, args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    description = (
        args.description_file.read_text(encoding="utf-8")
        if args.description_file
        else args.description
    )
    repo = args.repo
    if Path(repo).is_dir():
        repo = str(Path(repo).resolve())

    tracer = Tracer.for_run(cfg.trace.dir) if cfg.trace.enabled else Tracer.null()
    try:
        with tracer, tracer.span("review", repo=repo, base=args.base, head=args.head):
            mgr = WorkspaceManager.from_config(cfg.workspace)
            with mgr.checkout(repo, args.base, args.head, job_id=args.job_id) as ws:
                result = review_workspace(
                    ws,
                    llm,
                    cfg,
                    PRInfo(title=args.title, description=description),
                    tracer,
                    verifier_llm=_build_verifier(cfg, args, llm),
                )
    except DeepReviewError as e:
        print(f"review failed: {e}", file=sys.stderr)
        return 1

    payload = result.model_dump_json(indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    s = result.stats
    print(
        f"{len(result.findings)} findings ({len(result.dropped)} dropped) · "
        f"stop={s.get('stop_reason')} turns={s.get('turns', 0)} tools={s.get('tool_calls', 0)} "
        f"tokens={s.get('prompt_tokens', 0)}+{s.get('completion_tokens', 0)} "
        f"peak_ctx={s.get('peak_context_tokens', 0)} {s.get('duration_s', 0)}s"
        + (f" · trace {tracer.path}" if tracer.path else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
