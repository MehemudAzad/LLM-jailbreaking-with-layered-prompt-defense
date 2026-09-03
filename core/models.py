"""Model handles -- the shared contract between the attack battery and the defense stack.

Both halves code against `ModelHandle`. The real backend is implemented ONCE (jointly,
Week 11), then this file is frozen. Until then `backend = "fake"` gives every part of the
repo something callable so wiring can be tested end to end (`run_eval.py --dry-run`).

Roles (keys under [models.*] in config.toml):
    target             the victim model
    helper             helper LM for model-assisted attacks
    paraphraser        defense Layer 2
    perplexity_scorer  defense Layer 1
    judge              defense Layer 4 + automated grader
"""
from __future__ import annotations

import functools
import hashlib
from typing import Any

from core.config import CONFIG

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": str}


class ModelHandle:
    """Uniform interface over whatever backend serves a model."""

    def __init__(self, spec: dict[str, Any], role: str):
        self.spec = spec
        self.role = role
        self.name = spec.get("name", "<unset>")

    def generate(self, messages: list[Message] | str, *, prefill: str | None = None, **overrides) -> str:
        """Return the model's completion text only (no echo of the prompt)."""
        raise NotImplementedError

    def perplexity(self, text: str) -> float:
        """Log-perplexity of `text` under this model. Used by defense Layer 1."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.role}:{self.name}>"


class FakeModelHandle(ModelHandle):
    """Deterministic stand-in. No weights. Used by --dry-run and the wiring tests."""

    def generate(self, messages, *, prefill=None, **overrides) -> str:
        text = messages if isinstance(messages, str) else " ".join(m["content"] for m in messages)
        h = hashlib.sha256(text.encode()).hexdigest()[:8]
        if self.role == "judge":
            return "UNCLEAR"
        if self.role == "paraphraser":
            return f"[paraphrased#{h}] {text[:200]}"
        # target / helper: a refusal by default, so the fake pipeline behaves like a safe model
        return f"[fake:{self.name}#{h}] I can't help with that."

    def perplexity(self, text: str) -> float:
        n = max(len(text.split()), 1)
        return 50.0 + (int(hashlib.sha256(text.encode()).hexdigest(), 16) % 4000) / n


class TransformersModelHandle(ModelHandle):
    """HF Transformers backend.

    TODO(joint, Week 11) -- implement, then freeze this file:
      generate():   load once (cache on the class), apply the chat template, greedy-decode
                    with spec["temperature"] / spec["max_new_tokens"], honour `prefill` by
                    seeding the assistant turn, return ONLY the newly generated tokens.
      perplexity(): teacher-forced mean NLL over `text`; return exp(mean_nll). For the
                    windowed variant (defense Layer 1) return the minimum over token windows
                    of size config.defense.layer1_perplexity.window_size.
    """

    def generate(self, messages, *, prefill=None, **overrides) -> str:
        raise NotImplementedError("Implement TransformersModelHandle, then freeze core/models.py")

    def perplexity(self, text: str) -> float:
        raise NotImplementedError("Implement TransformersModelHandle, then freeze core/models.py")


_BACKENDS: dict[str, type[ModelHandle]] = {
    "fake": FakeModelHandle,
    "transformers": TransformersModelHandle,
    # "ollama": OllamaModelHandle,   # add if you go the Ollama route
}


@functools.lru_cache(maxsize=None)
def _load(role: str, force_fake: bool = False) -> ModelHandle:
    try:
        spec = dict(CONFIG["models"][role])
    except KeyError:
        raise KeyError(f"no [models.{role}] block in config.toml")
    backend = "fake" if force_fake else spec.get("backend", "fake")
    try:
        cls = _BACKENDS[backend]
    except KeyError:
        raise ValueError(f"unknown backend {backend!r} for model role {role!r}")
    return cls(spec, role)


def load_target(force_fake: bool = False) -> ModelHandle:
    return _load("target", force_fake)


def load_helper(force_fake: bool = False) -> ModelHandle:
    return _load("helper", force_fake)


def load_paraphraser(force_fake: bool = False) -> ModelHandle:
    return _load("paraphraser", force_fake)


def load_perplexity_scorer(force_fake: bool = False) -> ModelHandle:
    return _load("perplexity_scorer", force_fake)


def load_judge(force_fake: bool = False) -> ModelHandle:
    return _load("judge", force_fake)
