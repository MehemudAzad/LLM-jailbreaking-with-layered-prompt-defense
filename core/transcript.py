"""Append-only transcript logging. One directory per run under logs/<run-id>/:

    meta.json         run metadata (seed, pinned model names, python version)
    transcript.jsonl  one record per line; `type` in {config, trial, summary}

These files are the graded evidence for both reports and the demo. Never edit one
after the fact -- re-run instead.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from core.config import CONFIG, resolve


def new_run_id(tag: str = "") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{stamp}-{tag}-{short}" if tag else f"{stamp}-{short}"


def mirror_root() -> Path | None:
    """Where a finished run is copied so it outlives an ephemeral session.

    On Kaggle only /kaggle/working survives (it becomes the notebook Output), and the
    repo clone -- logs/ included -- is deleted at the start of the next run. Without a
    mirror the transcript is lost, and the transcript is the report's evidence.
    """
    explicit = os.environ.get("TRANSCRIPT_MIRROR") or CONFIG.get("paths", {}).get("mirror_dir")
    if explicit:
        return Path(explicit)
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working/artifacts")
    return None


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
        self._mirror()

    def _mirror(self) -> None:
        root = mirror_root()
        if root is None:
            return
        dest = root / self.run_id
        try:
            if dest.resolve() == self.dir.resolve():
                return
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.dir, dest, dirs_exist_ok=True)
            print(f"transcript mirrored -> {dest}")
        except Exception as exc:  # noqa: BLE001 -- a failed mirror must not lose the run
            print(f"WARNING mirror to {dest} failed ({exc}); download logs/{self.run_id} manually")

    def __enter__(self) -> "TranscriptLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
