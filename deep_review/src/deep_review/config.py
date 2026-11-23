"""Engine configuration.

Sources, lowest to highest precedence: defaults → TOML file → environment variables.
The API key is never stored in the config; only the *name* of the env var that holds it.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from deep_review.errors import ConfigError


def _default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "deep_review"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMConfig(_Section):
    base_url: str = ""
    """OpenAI-compatible endpoint: the internal gateway, or a dev endpoint for public code.
    Deliberately no default: a missing setting must fail, never fall back to a public API."""
    api_key_env: str = "DEEP_REVIEW_API_KEY"
    """Name of the env var (or .env entry) holding the key — never the key itself."""
    finder_model: str = "deepseek-flash"
    verifier_model: str = "deepseek-flash"
    """Empty → same as finder_model."""
    context_tokens: int = Field(100_000, gt=0)
    max_output_tokens: int = Field(8_000, gt=0)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    timeout_s: float = Field(120.0, gt=0)
    max_retries: int = Field(2, ge=0)
    structured_output: Literal["json_schema", "json_object"] = "json_schema"
    """How schema-bound output is requested. vLLM supports json_schema; the public DeepSeek
    API rejects it (400) and needs json_object."""

    def api_key(self, env: Mapping[str, str] | None = None) -> str:
        env = os.environ if env is None else env
        key = env.get(self.api_key_env, "")
        if not key:
            raise ConfigError(f"LLM API key is not set; export {self.api_key_env}")
        return key

    @property
    def effective_verifier_model(self) -> str:
        return self.verifier_model or self.finder_model


class BudgetConfig(_Section):
    shared_prefix_tokens: int = Field(40_000, gt=0)
    """PR summary + diff + impact map, shared by all specialists (prefix-cache friendly)."""
    tool_turn_tokens: int = Field(50_000, gt=0)
    """Room for tool calls and results inside one agent loop."""
    max_turns: int = Field(20, gt=0)
    max_parallel_llm_calls: int = Field(8, gt=0)


class WorkspaceConfig(_Section):
    cache_dir: Path = Field(default_factory=_default_cache_dir)
    git_timeout_s: float = Field(120.0, gt=0)
    fetch_timeout_s: float = Field(600.0, gt=0)
    lock_timeout_s: float = Field(900.0, gt=0)


class ToolLimits(_Section):
    max_file_bytes: int = Field(1_000_000, gt=0)
    read_window_lines: int = Field(200, gt=0)
    max_grep_results: int = Field(50, gt=0)
    max_tool_output_chars: int = Field(12_000, gt=0)


class SelectionConfig(_Section):
    max_changed_lines_per_file: int = Field(3_000, gt=0)
    max_line_length: int = Field(1_000, gt=0)
    """Added lines longer than this mark a file as minified/generated."""


class AggregateConfig(_Section):
    plausible_threshold: float = Field(0.8, ge=0.0, le=1.0)
    """Minimum confidence for a finding to be output. Calibrate on the eval set (PLAN §4.5);
    `eval run` sets it to 0 so `eval score` can sweep thresholds."""
    max_comments_per_pr: int = Field(10, gt=0)


class TraceConfig(_Section):
    enabled: bool = True
    dir: Path = Path("traces")


class ReviewConfig(_Section):
    finder_mode: Literal["recall", "precision"] = "recall"
    """recall: report every concretely-triggered defect with an honest confidence and let the
    threshold / verifier filter (PLAN §4.4). precision: the M1 baseline prompt."""


class PipelineConfig(_Section):
    mode: Literal["graph", "single"] = "graph"
    """graph: M3 specialists + verifiers (LangGraph). single: the M1 one-agent reviewer."""
    specialists: list[str] = Field(
        default_factory=lambda: ["logic", "contract", "concurrency_resource", "security"]
    )
    """The repo-rules specialist is added automatically when rules files are found."""
    specialist_max_turns: int = Field(12, gt=0)
    verify: bool = True
    verifier_max_turns: int = Field(8, gt=0)
    max_candidates_to_verify: int = Field(30, gt=0)
    max_concurrency: int = Field(6, gt=0)
    """Parallel agent loops inside one review (LangGraph max_concurrency)."""
    dedupe_line_window: int = Field(3, ge=0)
    max_rules_tokens: int = Field(4_000, gt=0)


class EvalConfig(_Section):
    judge_model: str = ""
    """Model that judges finding↔ground-truth matches. Empty → verifier model. Prefer a model
    other than the finder's, so the engine is not grading its own output."""
    runs_dir: Path = Path("eval_runs")
    parallel_cases: int = Field(4, gt=0)
    line_window: int = Field(3, ge=0)


class Config(_Section):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    tools: ToolLimits = Field(default_factory=ToolLimits)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    aggregate: AggregateConfig = Field(default_factory=AggregateConfig)
    trace: TraceConfig = Field(default_factory=TraceConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @model_validator(mode="after")
    def _budget_fits_context(self) -> Config:
        b, llm = self.budgets, self.llm
        used = b.shared_prefix_tokens + b.tool_turn_tokens + llm.max_output_tokens
        if used > llm.context_tokens:
            raise ValueError(
                f"budgets exceed the context window: shared_prefix ({b.shared_prefix_tokens}) + "
                f"tool_turns ({b.tool_turn_tokens}) + max_output ({llm.max_output_tokens}) = "
                f"{used} > context_tokens ({llm.context_tokens})"
            )
        return self


# env var → (section, key)
ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "DEEP_REVIEW_BASE_URL": ("llm", "base_url"),
    "DEEP_REVIEW_MODEL": ("llm", "finder_model"),
    "DEEP_REVIEW_VERIFIER_MODEL": ("llm", "verifier_model"),
    "DEEP_REVIEW_CONTEXT_TOKENS": ("llm", "context_tokens"),
    "DEEP_REVIEW_STRUCTURED_OUTPUT": ("llm", "structured_output"),
    "DEEP_REVIEW_CACHE_DIR": ("workspace", "cache_dir"),
    "DEEP_REVIEW_TRACE_DIR": ("trace", "dir"),
    "DEEP_REVIEW_JUDGE_MODEL": ("eval", "judge_model"),
    "DEEP_REVIEW_FINDER_MODE": ("review", "finder_mode"),
    "DEEP_REVIEW_PIPELINE": ("pipeline", "mode"),
}


def load_env_file(path: Path | str) -> dict[str, str]:
    """Minimal .env reader: KEY=VALUE lines, # comments, optional surrounding quotes."""
    values: dict[str, str] = {}
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.removeprefix("export ").partition("=")
        if not sep or not key.strip().isidentifier():
            raise ConfigError(f"{path}:{n}: expected KEY=VALUE")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_config(
    path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    env_file: Path | str | None = ".env",
) -> Config:
    """With `env=None`, entries from `env_file` are added to os.environ without overriding
    variables that are already set (so `LLMConfig.api_key()` sees them). An explicit `env`
    mapping is used as-is and no file is read — that keeps tests hermetic."""
    if env is None:
        if env_file and Path(env_file).is_file():
            for key, value in load_env_file(env_file).items():
                os.environ.setdefault(key, value)
        env = os.environ
    data: dict[str, Any] = {}
    path = path or env.get("DEEP_REVIEW_CONFIG")
    if path:
        try:
            data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise ConfigError(f"cannot read config {path}: {e}") from e
    for var, (section, key) in ENV_OVERRIDES.items():
        if env.get(var):
            data.setdefault(section, {})[key] = env[var]
    try:
        return Config.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"invalid configuration:\n{e}") from e
