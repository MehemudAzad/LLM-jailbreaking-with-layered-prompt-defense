"""Defense-stack contract. FROZEN after Week 10.

Member B: fill in the four layer files, don't edit this. A layer subclasses
`DefenseLayer`, sets `name` / `stage`, implements `process(ctx) -> ctx`, and records a
`Verdict` via `ctx.record(...)`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Stage(str, Enum):
    PRE = "pre"     # runs on the prompt, before the target model
    POST = "post"   # runs on the target's response


class Action(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    TRANSFORM = "transform"


@dataclass
class Verdict:
    layer: str
    action: Action
    reason: str = ""
    score: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class DefenseContext:
    goal_id: str
    attack: str
    original_prompt: str
    prompt: str                                  # current prompt (PRE layers may transform it)
    system: str = ""                             # system prompt handed to the target
    prefill: str | None = None
    response: str | None = None                  # filled in by the harness after the target runs
    blocked: bool = False
    blocked_by: str | None = None
    verdicts: list[Verdict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def record(self, verdict: Verdict) -> None:
        self.verdicts.append(verdict)
        if verdict.action == Action.BLOCK and not self.blocked:
            self.blocked = True
            self.blocked_by = verdict.layer


class DefenseLayer(ABC):
    name: str = "unnamed"
    stage: Stage = Stage.PRE
    source: str = ""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.force_fake = bool(self.config.get("_force_fake", False))

    @abstractmethod
    def process(self, ctx: DefenseContext) -> DefenseContext:
        """Inspect/transform ctx in place, record a Verdict, return ctx."""
