from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from deep_review.config import load_config
from deep_review.errors import ConfigError
from deep_review.trace import Tracer


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_spans_nest_and_record_errors(tmp_path: Path) -> None:
    with Tracer.for_run(tmp_path, run_id="r1") as t, t.span("outer", pr=7) as outer:
        t.event("note", msg="hi")
        with pytest.raises(ValueError), t.span("inner"):
            raise ValueError("boom")
    recs = _records(tmp_path / "r1.jsonl")
    assert [r["kind"] for r in recs] == ["span_start", "note", "span_start", "span_end", "span_end"]
    assert [r["seq"] for r in recs] == list(range(5))
    assert recs[0]["data"] == {"name": "outer", "parent": None, "pr": 7}
    assert recs[1]["span"] == outer
    assert recs[2]["data"]["parent"] == outer
    assert recs[3]["data"]["status"] == "error" and "boom" in recs[3]["data"]["error"]
    assert recs[4]["data"]["status"] == "ok" and recs[4]["span"] == outer


def test_concurrent_writes_stay_line_atomic(tmp_path: Path) -> None:
    t = Tracer(tmp_path / "c.jsonl")
    threads = [
        threading.Thread(target=lambda i=i: [t.event("e", i=i, j=j) for j in range(200)])
        for i in range(8)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    t.close()
    recs = _records(tmp_path / "c.jsonl")
    assert len(recs) == 1600 and len({r["seq"] for r in recs}) == 1600


def test_null_tracer_is_silent() -> None:
    t = Tracer.null()
    with t.span("x"):
        t.event("y")


def test_config_defaults_and_overrides(tmp_path: Path) -> None:
    cfg = load_config(env={})
    assert cfg.llm.context_tokens == 100_000
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text('[llm]\nfinder_model = "from-file"\n[aggregate]\nmax_comments_per_pr = 3\n')
    cfg = load_config(
        cfg_file, env={"DEEP_REVIEW_MODEL": "from-env", "DEEP_REVIEW_BASE_URL": "http://x/v1"}
    )
    assert cfg.llm.finder_model == "from-env"
    assert cfg.llm.effective_verifier_model == "from-env"  # follows the finder unless set
    assert cfg.llm.base_url == "http://x/v1"
    assert cfg.aggregate.max_comments_per_pr == 3


def test_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="exceed the context window"):
        load_config(env={"DEEP_REVIEW_CONTEXT_TOKENS": "50000"})
    bad = tmp_path / "bad.toml"
    bad.write_text("[llm]\nunknown_key = 1\n")
    with pytest.raises(ConfigError):
        load_config(bad, env={})
    with pytest.raises(ConfigError, match="DEEP_REVIEW_API_KEY"):
        load_config(env={}).llm.api_key(env={})
