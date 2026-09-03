"""Append-only transcript logging. One directory per run under logs/<run-id>/:

    meta.json         run metadata (seed, pinned model names, python version)
    transcript.jsonl  one record per line; `type` in {config, trial, summary}

These files are the graded evidence for both reports and the demo. Never edit one
after the fact -- re-run instead.
"""
from __future__ import annotations

import json
import platform
import time
import uuid
from pathlib import Path
from typing import Any

from core.config import CONFIG, resolve


def new_run_id(tag: str = "") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{stamp}-{tag}-{short}" if tag else f"{stamp}-{short}"


class TranscriptLogger:
    def __init__(self, run_id: str | None = None, tag: str = ""):
        self.run_id = run_id or new_run_id(tag)
        self.dir: Path = resolve(CONFIG["paths"]["logs_dir"]) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "transcript.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")
        self._write_meta()

    def _write_meta(self) -> None:
        meta = {
            "run_id": self.run_id,
            "ts": time.time(),
            "seed": CONFIG.get("seed"),
            "models": {role: spec.get("name") for role, spec in CONFIG.get("models", {}).items()},
            "python": platform.python_version(),
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        record.setdefault("ts", time.time())
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "TranscriptLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
