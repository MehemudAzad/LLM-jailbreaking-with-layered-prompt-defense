"""Frozen prompt sets. Member A owns the .jsonl files; this loader is shared.

harmful_behaviors.jsonl -- one line per record:
    {"id": "hb_0001", "category": "...", "goal": "<harmful instruction>",
     "target_prefix": "Sure, here is", "source": "AdvBench"}

benign_prompts.jsonl -- one line per record:
    {"id": "bn_0001", "text": "<ordinary request>"}

Lines starting with `//` and blank lines are ignored (so the files can carry headers).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from core.config import CONFIG, resolve


@dataclass(frozen=True)
class Goal:
    id: str
    goal: str
    category: str = "unspecified"
    target_prefix: str = "Sure, here is"  # affirmative-response prefix (Wei et al.)
    source: str = ""


@dataclass(frozen=True)
class BenignPrompt:
    id: str
    text: str


def _read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(resolve(path), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                yield json.loads(line)


def load_harmful() -> list[Goal]:
    return [
        Goal(
            id=r["id"],
            goal=r["goal"],
            category=r.get("category", "unspecified"),
            target_prefix=r.get("target_prefix", "Sure, here is"),
            source=r.get("source", ""),
        )
        for r in _read_jsonl(CONFIG["datasets"]["harmful"])
    ]


def load_benign() -> list[BenignPrompt]:
    return [BenignPrompt(id=r["id"], text=r["text"]) for r in _read_jsonl(CONFIG["datasets"]["benign"])]
