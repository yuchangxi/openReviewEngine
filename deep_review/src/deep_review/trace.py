"""JSONL tracing (PLAN §3): one file per run, one JSON object per line, thread-safe.

Record shape: {"ts", "run", "seq", "kind", "span", "data"}. Spans nest via a ContextVar.
Never pass secrets in `data`.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

_current_span: ContextVar[str | None] = ContextVar("deep_review_span", default=None)


class Tracer:
    def __init__(self, path: Path | str | None, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._seq = itertools.count()
        self._fh: TextIO | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    @classmethod
    def for_run(cls, trace_dir: Path | str, run_id: str | None = None) -> Tracer:
        run_id = run_id or uuid.uuid4().hex[:12]
        return cls(Path(trace_dir) / f"{run_id}.jsonl", run_id)

    @classmethod
    def null(cls) -> Tracer:
        return cls(None)

    def event(self, kind: str, **data: Any) -> None:
        if self._fh is None:
            return
        with self._lock:
            record = {
                "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "run": self.run_id,
                "seq": next(self._seq),
                "kind": kind,
                "span": _current_span.get(),
                "data": data,
            }
            self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()

    @contextmanager
    def span(self, name: str, **data: Any) -> Generator[str]:
        span_id = uuid.uuid4().hex[:8]
        parent = _current_span.get()
        token = _current_span.set(span_id)
        self.event("span_start", name=name, parent=parent, **data)
        t0 = time.perf_counter()
        status, error = "ok", None
        try:
            yield span_id
        except BaseException as e:
            status, error = "error", f"{type(e).__name__}: {e}"
            raise
        finally:
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            self.event("span_end", name=name, status=status, error=error, duration_ms=duration_ms)
            _current_span.reset(token)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> Tracer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
