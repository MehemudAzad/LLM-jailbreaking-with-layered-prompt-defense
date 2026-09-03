"""Attack-battery contract. FROZEN after Week 10.

Member A: add one file per technique, don't edit this. A technique is a class that
subclasses `AttackTechnique`, sets `name` / `category`, implements `apply()`, and is
decorated with `@register` (from `attacks`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.models import ModelHandle

CATEGORIES = ("encoding", "injection", "persona", "assisted", "combination", "control")


@dataclass
class AttackOutput:
    """What a technique produces from a raw harmful `goal` string."""

    prompt: str                                  # user-turn text to send to the target
    system: str | None = None                    # system prompt the attack wants (None -> pipeline default)
    prefill: str | None = None                   # forced assistant prefix, if the technique uses one
    decode: str | None = None                    # "base64" / "rot13" / ... -> decode the reply before judging
    metadata: dict[str, Any] = field(default_factory=dict)


class AttackTechnique(ABC):
    name: str = "unnamed"
    category: str = "control"
    #: True if apply() needs a helper LM (the auto_* techniques)
    uses_helper_lm: bool = False
    #: one-line provenance, e.g. "Wei et al. 2023, Table 1"
    source: str = ""

    @abstractmethod
    def apply(self, goal: str, *, helper: ModelHandle | None = None) -> AttackOutput:
        """Wrap a raw harmful `goal` into the attack prompt."""
