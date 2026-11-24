# Evaluation datasets

One JSON object per line, schema `EvalCase` in `src/deep_review/evaluation/dataset.py`.

| File | Source | Cases | Ground truth |
|---|---|---|---|
| `aacr_v2_java_ts_js.jsonl` | AACR-Bench v2, Java + TypeScript + JavaScript subset | 52 (24 Java, 19 TS, 9 JS) | 342 issues: 228 defect, 8 security, 40 performance, 66 maintainability |
| `qodo_v1.jsonl` | Qodo Code Review Benchmark 1.0 (all) | 100 (26 Python, 16 TS, 13 JS, 13 Swift, 13 Rust, 10 C#, 9 C) | 580 injected issues: 309 functional bugs (`defect`), 271 repository-rule violations (`maintainability`) |

**Primary benchmark: `qodo_v1.jsonl`.** Its issues were injected on purpose and double-validated,
with naturally occurring issues found during validation added, so its ground truth is close to
complete and measured precision is meaningful. AACR-Bench ground truth is review comments and
misses many real defects (a manual audit found most "false positives" on flagged lines were
real bugs it does not list), so AACR precision is only a lower bound.

## Attribution

`aacr_v2_java_ts_js.jsonl` is derived from **AACR-Bench** by Alibaba
(https://github.com/alibaba/aacr-bench), licensed under the Apache License 2.0.
Changes: converted to the `EvalCase` format, restricted to three languages, PR titles and
descriptions added from the public GitHub API, categories normalized. Ground-truth text is
unchanged.

Regenerate:

```bash
git clone https://github.com/alibaba/aacr-bench /path/to/aacr-bench
uv run deep_review eval import-aacr /path/to/aacr-bench/dataset/v2/positive_samples.json \
  --out datasets/aacr_v2_java_ts_js.jsonl --languages Java,TypeScript,JavaScript
```

`qodo_v1.jsonl` is derived from **Qodo PR-Review-Bench** (https://huggingface.co/datasets/Qodo/PR-Review-Bench),
licensed under the MIT License. Changes: converted to the `EvalCase` format; base/head SHAs
resolved from the public GitHub API for the PRs in https://github.com/agentic-review-benchmarks;
the "Code Review Bench PR #N - " title prefix removed; rule violations mapped to
`maintainability` (scope `rule: <name>`), other issues to `defect`; up to 400 characters of the
problematic code snippet appended to the issue text. Issue titles/descriptions are unchanged.

Regenerate:

```bash
mkdir -p datasets/raw/qodo && curl -sSfL -o datasets/raw/qodo/bench.jsonl \
  https://huggingface.co/datasets/Qodo/PR-Review-Bench/resolve/main/git_code_review_bench_100_w_open_prs.jsonl
uv run deep_review eval import-qodo datasets/raw/qodo/bench.jsonl --out datasets/qodo_v1.jsonl
```

## Caveats

Qodo:
- Report the `all` subset to compare with Qodo's published table (it scores rule violations
  too); `bugs` = functional bugs only. Each fork has an AGENTS.md with the repository rules
  at the base commit, which our `rules` specialist picks up.
- 37 issues (6%) have no file/line; they are offered to the judge for a finding in any file
  and are skipped by the judge-free line-window metric.
- The issues are LLM-injected (critics note they may not match real defect distributions) and
  the benchmark is vendor-built; no Java cases.
- Published scores use Qodo's own judge, so compare trends, not absolute numbers.

AACR:

- 4 cases cannot be checked out: their head commits were force-pushed away and GitHub no
  longer serves them (`upload-pack: not our ref`): keycloak-35645, -36457, -37465, -37504.
  Using the current PR head instead would misalign the ground-truth line numbers, so runs
  report them as missing and exclude them from every denominator (48 scorable cases).

- AACR-Bench is Alibaba's own benchmark; their OCR reviewer (our tier 1) may be favoured on it.
- About three quarters of the ground-truth comments are model-generated and then human-verified.
  Ground truth is incomplete, so measured precision is a lower bound.
- These are public open-source PRs, not banking code. They measure general review ability.
  A banking-specific set (money arithmetic, transactions, idempotency, authorization) is still
  needed once intranet history is available.
