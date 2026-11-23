# deep_review

Tier-2 deep code review engine. Design and milestones: [PLAN.md](PLAN.md).

## Development

```bash
uv sync                  # create .venv with runtime + dev deps
uv run pytest            # offline unit tests
uv run pytest -m network # also clones a public GitHub repo PR (needs internet)
uv run ruff check .
```

## Review one PR

```bash
uv run deep_review review --repo https://github.com/OWNER/REPO.git \
  --base main --head refs/pull/123/head --title "..." --out result.json
```

`--record file.jsonl` saves every model exchange; `--replay file.jsonl` reruns offline.

## Pipelines

- `graph` (default, M3): build_context → 4 specialists in parallel (logic, contract/callers,
  concurrency/resources, security; plus repo rules when AGENTS.md/CLAUDE.md/REVIEW.md exist)
  → dedupe + position check → one fresh-context verifier per candidate → aggregate.
  Orchestrated with LangGraph; LangSmith tracing is forced off.
- `single` (M1): one bounded agent. `DEEP_REVIEW_PIPELINE=single`.

Output keeps CONFIRMED findings and PLAUSIBLE ones at or above `aggregate.plausible_threshold`.

## Evaluation (PLAN §6)

```bash
uv run deep_review eval run   --dataset datasets/aacr_v2_java_ts_js.jsonl --run-id NAME
uv run deep_review eval score --dataset datasets/aacr_v2_java_ts_js.jsonl --run-id NAME
uv run deep_review eval compare BASELINE NAME
```

Runs are resumable and land in `eval_runs/NAME/`: per-case results, traces, recordings,
`score.json`, `report.md` and `judge_sample.md` (verify the judge by hand). Precision counts a
finding as correct if it matches any verified reference. Recall is reported for `all`, `bugs`
(defect + security + performance) and `defect_security`. See `datasets/README.md`.

## Configuration

Defaults are in `src/deep_review/config.py`. Override with a TOML file
(`DEEP_REVIEW_CONFIG=path/to/config.toml`) and/or environment variables:

| Variable | Sets |
|---|---|
| `DEEP_REVIEW_BASE_URL` | `llm.base_url` — OpenAI-compatible endpoint |
| `DEEP_REVIEW_MODEL` / `DEEP_REVIEW_VERIFIER_MODEL` | finder / verifier model names |
| `DEEP_REVIEW_API_KEY` | API key (name configurable via `llm.api_key_env`; never stored in config) |
| `DEEP_REVIEW_CONTEXT_TOKENS` | `llm.context_tokens` (default 100k) |
| `DEEP_REVIEW_CACHE_DIR` | bare-repo cache and worktrees |
| `DEEP_REVIEW_TRACE_DIR` | JSONL trace output |
| `DEEP_REVIEW_STRUCTURED_OUTPUT` | `json_schema` (vLLM) or `json_object` (public DeepSeek API) |
| `DEEP_REVIEW_JUDGE_MODEL` | eval judge model (default: verifier model) |
| `DEEP_REVIEW_PIPELINE` | `graph` (M3, default) or `single` (M1) |
| `DEEP_REVIEW_FINDER_MODE` | `recall` (default) or `precision` (M1 baseline prompt) |

Local settings go in `.env` (gitignored; see `.env.example`). Real environment variables win.
Only public open-source code may be sent to an endpoint outside the intranet.
