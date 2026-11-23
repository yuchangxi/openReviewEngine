# Evaluation datasets

One JSON object per line, schema `EvalCase` in `src/deep_review/evaluation/dataset.py`.

| File | Source | Cases | Ground truth |
|---|---|---|---|
| `aacr_v2_java_ts_js.jsonl` | AACR-Bench v2, Java + TypeScript + JavaScript subset | 52 (24 Java, 19 TS, 9 JS) | 342 issues: 228 defect, 8 security, 40 performance, 66 maintainability |

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

## Caveats

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
