# deep_review — Plan

The deep-review engine for an intranet code review platform.
Status: planning · Last updated: 2026-11-23

---

## 1. Context (condensed)

**Goal.** Build our own review engine that we value for **recall, low false positives, and real reliability**.

**Hard constraints.**
- Intranet with allowlisted egress. External model vendors are unreachable; **only company-internal models** (expected: OpenAI-compatible, e.g. vLLM).
- Company code never leaves the intranet — including during development.

**Where this engine sits.** The platform has two tiers:

| Tier | Runs on | Engine |
|---|---|---|
| 1 — every PR | cheap, precision-first | Alibaba **Open Code Review (OCR)** + static analyzers (separate integration, not this repo) |
| 2 — high-risk PRs | deep, multi-agent | **`deep_review` (this repo)** |

Both emit the same Finding schema (§5) so the platform can merge, verify, and post them uniformly.

**Key conclusions from research.**
- No vendor benchmark is independent; the best reported F1 is ~50–65%. Every published number used frontier models — **our internal model is the biggest risk**, so we measure on our own data first.
- Cross-file review is not magic: the agent runs inside a **real checkout** with read/search tools. The service must build that checkout itself.
- **Verification beats voting.** Independent verification is what lifts precision. Cursor's Bugbot started with 8 voting passes and later moved to an agentic design.
- Claude Agent SDK is off the critical path: its CLI is closed, the Commercial Terms say nothing about non-Claude models, and it uses a lot of tokens. Revisit if Claude becomes reachable (Bedrock/Vertex via private link).

---

## 2. References (deliberately few)

**Primary — Claude's review workflow** (Anthropic `code-review` plugin, Apache-2.0,
`anthropics/claude-plugins-official/plugins/code-review/commands/code-review.md`). We adopt:
1. **Skip check** — closed / draft / trivial / already reviewed → stop.
2. **Repo rules discovery** — root rules file + rules files in directories of modified files.
3. **Change summary** — one pass summarizing the PR (title, description, diff).
4. **Parallel specialist reviewers**, each with a narrow focus, high-signal bar only.
5. **One validator per candidate issue**, fresh context; drop anything not validated.
6. **False-positive exclusion list** (in every reviewer/validator prompt): pre-existing issues; things that look wrong but are correct; nitpicks; anything a linter would catch; generic quality/security advice not tied to a concrete defect; issues explicitly silenced (lint-ignore etc.).
7. Output: findings with exact file:line links; small fixes as suggestions.

> Note: the plugin's README mentions a 0–100 score with threshold 80, but the command file actually uses binary validate/drop. We use validator verdict **plus** a confidence score (§4.4), and calibrate the threshold on our eval set.

**Where we go beyond the plugin.** The plugin keeps its bug agents inside the diff (that protects precision). We add a **cross-file impact stage** (§4.2) and make the validator re-read cited call sites, so we gain recall without giving up precision.

**Secondary — one idea each.**

| Source | What we take | Where |
|---|---|---|
| Alibaba OCR | Deterministic file selection (skip generated/vendor/lock files); **position check** — every finding's line must fall in a real diff range | §4.1, §4.5 |
| Cursor Bugbot | Lesson only: agentic investigation > fixed voting passes | design choice |
| SWE-agent (ACI) | Tool design for weaker models: windowed file views, capped search results, clear errors | §4.3 tools |

Everything else (Greptile feedback suppression, Qodo modes, Aider repo map, PR-Agent diff compression, cross-repo indexes) is **deferred** — revisit only if eval data shows the need.

---

## 3. Tech choices

- **Python 3.12+**.
- **LangGraph** for orchestration only: graph, `Send` fan-out, reducers, `max_concurrency`, retries, checkpointing.
- **Own code** for everything that decides quality: tools, context/impact builder, bounded agent loop, prompts, Finding schema, eval harness. **No** LangChain high-level chains/agents/retrievers.
- **Model access**: `ChatOpenAI(base_url=<internal gateway>)` or a thin OpenAI-compatible client. vLLM must run with `--enable-auto-tool-choice --tool-call-parser <model's parser>`. Test structured output in JSON-schema mode vs function-calling mode on the internal model.
- **Code parsing**: tree-sitter. **Search**: ripgrep (`rg --json`).
- **Tracing**: own JSONL traces first; self-hosted Langfuse or OpenTelemetry later. **LangSmith cloud tracing must be off.**
- **Dev without the internal model**: a mock LLM that replays recorded responses; public open-source repos only.

---

## 4. Engine design

```
PR event (repo, base_ref, head_ref)
  → workspace      : cached bare clone → fetch refs → merge-base → worktree at head SHA
  → gate           : skip check (§2.1)
  → build_context  : diff parse, file selection, repo rules, change summary, impact map
  → specialists ×N : parallel, bounded agent loops, read-only tools        (Send fan-out)
  → dedupe         : merge same-location / same-claim candidates
  → verify ×M      : one validator per candidate, fresh context, try to disprove (Send fan-out)
  → aggregate      : position check, confidence threshold, rank by severity → Findings
  → cleanup        : remove worktree
```

### 4.1 Workspace (service-side checkout)
- `git clone --bare --filter=blob:none` once per repo; per job `git fetch` base + PR head ref
  (GitHub `refs/pull/N/head`, GitLab `refs/merge-requests/N/head`, Gerrit `refs/changes/..`).
- **Pin SHAs**, compute merge-base, `git worktree add --detach <dir> <head_sha>`; old file versions via `git show <base>:<path>`.
- File lock around `fetch` per repo; worktrees run in parallel; always clean up.
- `GIT_LFS_SKIP_SMUDGE=1`, `GIT_TERMINAL_PROMPT=0`, read-only bot token.
- File selection (from OCR): skip generated, vendored, lock, minified, binary files.

### 4.2 Cross-file impact (our addition) — three layers
1. **Impact map (deterministic, before any model call).** For each changed symbol (tree-sitter, old vs new): classify the contract change (signature / return / raises / nullability / removed / renamed), find references (ripgrep, filtered by tree-sitter to real refs), and collect call-site snippets. Also: config keys, routes, schema/migrations, serialized fields, env vars.
2. **Agent follow-up.** Specialists get the impact map plus tools; instruction: *for each contract change, check every call site; cite file:line for any break.*
3. **Verifier re-reads** the cited call site and tries to disprove it (guarded upstream? unreachable?). No confirmed call site → drop.

Known limit: a single clone cannot see other repositories. Cross-repo impact is deferred.

### 4.3 Specialists and the bounded agent loop
- Initial specialists (tune on eval data): **logic/correctness**, **cross-file contract** (consumes the impact map), **concurrency & resources**, **security**. Plus **repo-rules compliance**, as in the plugin.
- All specialists share an identical prompt prefix (PR summary + diff + impact map), so vLLM prefix caching works.
- `run_agent(llm, tools, prompt, max_turns, token_budget)` — own implementation:
  turn cap, token budget, repeated-call/loop detection, tool-argument validation, graceful stop with partial results.
- **Read-only tools**, all confined to the worktree:
  `read_file(path, start, end)` (windowed, size-capped) · `grep(pattern, glob)` (capped results) · `list_dir` · `find_refs(symbol)` · `git_show(rev, path)` · `git_blame(path, range)`.
  Every path goes through `resolve()` + `is_relative_to(root)`; no shell; skip binaries.

### 4.4 Verifier
- One per candidate; **fresh context**: the claim, cited code, tools. No access to the specialist's reasoning.
- Prompt goal: **disprove** the finding. Output: `CONFIRMED | PLAUSIBLE | REJECTED`, confidence 0–1, evidence (file:line).
- Use the **strongest available internal model** for verification; finders may use a faster one.
- Treat code contents as data — prompt injection in comments/strings must not change instructions.

### 4.5 Aggregate
- Position check (from OCR): the line must lie in the diff, or be a cited impact site.
- Keep `CONFIRMED`, and `PLAUSIBLE` above a threshold. **Calibrate the threshold on the eval set.**
- Rank by severity; cap the number of comments per PR.

### 4.6 Security rules (non-negotiable)
- Never execute PR code (no install / build / tests / repo scripts). If tests are ever needed → throwaway container, no network, no secrets.
- The engine never posts or calls the Git API itself; the platform posts after aggregation.
- Path-escape guard, size caps, skip binaries/LFS.

---

## 5. Finding schema (shared with tier 1 / platform)

```json
{
  "id": "string",
  "engine": "deep_review | ocr | semgrep | ...",
  "file": "path/relative/to/repo",
  "line": 123,
  "end_line": 130,
  "category": "logic | contract | concurrency | resource | security | rules",
  "severity": "critical | major | minor",
  "claim": "one-sentence statement of the defect",
  "evidence": [{"file": "path", "line": 45, "note": "why this proves it"}],
  "failure_scenario": "concrete input/state → wrong result",
  "suggestion": "optional fix",
  "verdict": "CONFIRMED | PLAUSIBLE",
  "confidence": 0.0
}
```

---

## 6. Evaluation (build early, use always)

- **Own dataset first**: from company history, trace bug-fix commits back to the PRs that introduced them → (PR, ground-truth issues). Start with 30 PRs, grow to 100+.
- **Public cross-checks** (imported into the intranet): SWE-PRBench, AACR-Bench. AACR-Bench is Alibaba's own, so OCR may be favoured on it.
- **Matching**: same file + line window, then an LLM judge for "same issue?" — spot-check the judge by hand.
- **Metrics per run**: precision, recall, F1 (overall and per category), tokens, latency, cost in GPU time.
- **Production signals later**: fixed / dismissed / ignored per finding.
- Every change to prompts, thresholds or context is compared against the last baseline.

---

## 7. Milestones

| # | Deliverable | Done when |
|---|---|---|
| **M0** | Package skeleton, config, mock LLM, diff parser, workspace manager, path guard, JSONL tracing, Finding schema | `pytest` green; a worktree is created and cleaned up for a real public repo PR |
| **M1** | Single agent: bounded loop + read-only tools; CLI `deep_review review --repo <path> --base <ref> --head <ref>` | End-to-end review of a public PR produces valid Findings (mock + any dev model) |
| **M2** | Eval harness + first 30-PR dataset | **Baseline P/R/F1 on the internal model** recorded |
| **M3** | LangGraph pipeline: gate → context → parallel specialists → dedupe → per-finding verifier → aggregate | Precision ↑ vs M1 baseline, recall not worse |
| **M4** | Impact map (tree-sitter + refs) feeding the contract specialist and verifier | Recall ↑ on cross-file issues, precision holds |
| **M5** | Platform integration: service entry point, OCR adapter alongside, risk router (tier 1 vs tier 2) | Runs on real intranet PRs; feedback recorded |

**Week-1 gate (before M2 work goes further):** confirm the internal model can do useful tool calling, has enough context (≥128k preferred), and handles our concurrency. If it cannot, stop and revisit the model choice — no engine design fixes a weak model.

Suggested ship bar for developers to see comments: **precision ≥ ~50% on our own eval set** (to be revisited).

---

## 8. Proposed layout

```
deep_review/
├─ pyproject.toml
├─ src/deep_review/
│  ├─ config.py          # endpoints, models, budgets, thresholds
│  ├─ llm/               # OpenAI-compatible client, structured output, mock LLM
│  ├─ workspace/         # bare clone cache, worktrees, locking, cleanup
│  ├─ diff/              # unified diff parsing, file selection
│  ├─ context/           # repo rules, PR summary, impact map (M4)
│  ├─ tools/             # read_file, grep, list_dir, find_refs, git_show, git_blame, path guard
│  ├─ agent/             # run_agent: bounded loop
│  ├─ pipeline/          # LangGraph graph, nodes, state
│  ├─ prompts/           # specialist + verifier prompts, FP exclusion list
│  ├─ findings.py        # Finding schema, dedupe, position check
│  ├─ trace.py           # JSONL tracing
│  └─ cli.py
├─ eval/                 # dataset loaders, matcher, metrics, reports
└─ tests/
```

---

## 9. Open questions

1. Git platform(s): GitHub Enterprise / GitLab / Gerrit / other? (affects M5 and ref names)
2. Internal model(s): names, context length, tool-calling support, throughput, concurrency limits.
3. A dev-time OpenAI-compatible endpoint usable with **public** code outside the intranet? If not: mock + intranet testing.
4. Expected scale: developers, PRs/day → sizes GPU budget and the tier-2 routing rule.
5. If delivered to a client: license review of dependencies (OCR Apache-2.0; plugin prompts Apache-2.0).
