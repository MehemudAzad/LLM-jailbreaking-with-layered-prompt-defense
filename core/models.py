"""Model handles -- the shared contract between the attack battery and the defense stack.

Both halves code against `ModelHandle`. `backend = "fake"` gives every part of the repo
something callable with no weights (`run_eval.py --dry-run`, the wiring tests).
`backend = "transformers"` is the real path -- validated on Kaggle, see
notebooks/m1_model_backend.ipynb.

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
    """Hugging Face Transformers backend (milestone M1).

    Loads the model once per (name, revision), cached on the class. Handles both plain
    text models (AutoModelForCausalLM + AutoTokenizer) and text+vision models used
    text-only (AutoModelForImageTextToText + AutoProcessor -- e.g. google/gemma-3-4b-it).

    All heavy imports are lazy, so `backend = "fake"` / `--dry-run` never touch torch.

    `perplexity()` is still a stub -- it lands in the defense Layer-1 milestone (it needs
    a different model, `perplexity_scorer` / gpt2-large, not the target).
    """

    _CACHE: dict = {}

    def _revision(self):
        rev = self.spec.get("revision")
        return None if rev in (None, "", "PIN-ME") else rev

    def _resolve_dtype(self):
        import torch

        want = str(self.spec.get("dtype", "auto")).lower()
        if want in ("auto", ""):
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16      # RTX 6000 etc.
            return torch.float16           # T4 (Turing has no bf16 tensor cores)
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}.get(want, torch.float16)

    def _bundle(self):
        key = (self.name, self._revision())
        cached = TransformersModelHandle._CACHE.get(key)
        if cached is not None:
            return cached

        import transformers as tf

        load_kw = dict(revision=self._revision(), torch_dtype=self._resolve_dtype(), device_map="auto")
        try:
            tokenizer = tf.AutoTokenizer.from_pretrained(self.name, revision=self._revision())
            model = tf.AutoModelForCausalLM.from_pretrained(self.name, **load_kw)
            bundle = (model.eval(), tokenizer, None, "causal")
        except Exception:  # noqa: BLE001 -- arch isn't a plain CausalLM (Gemma 3 4B, other VLMs)
            processor = tf.AutoProcessor.from_pretrained(self.name, revision=self._revision())
            model = tf.AutoModelForImageTextToText.from_pretrained(self.name, **load_kw)
            bundle = (model.eval(), None, processor, "vlm")

        TransformersModelHandle._CACHE[key] = bundle
        return bundle

    @staticmethod
    def _fold_system(messages: list[Message]) -> list[Message]:
        """Fold any system turn into the next user turn -- Gemma's chat template (and some
        others) has no system slot. Models with a real system role lose the separation;
        acceptable for the target, and the defense writeup should note it."""
        out: list[Message] = []
        sys_buf = ""
        for m in messages:
            if m["role"] == "system":
                sys_buf += m["content"].strip() + "\n\n"
            elif m["role"] == "user":
                out.append({"role": "user", "content": (sys_buf + m["content"]) if sys_buf else m["content"]})
                sys_buf = ""
            else:
                out.append(dict(m))
        if sys_buf:
            out.insert(0, {"role": "user", "content": sys_buf.strip()})
        return out

    def _gen_kwargs(self, overrides: dict) -> dict:
        temperature = float(overrides.get("temperature", self.spec.get("temperature", 0.0)))
        kw = dict(
            max_new_tokens=int(overrides.get("max_new_tokens", self.spec.get("max_new_tokens", 512))),
            do_sample=temperature > 0.0,
        )
        if kw["do_sample"]:
            kw["temperature"] = temperature
            kw["top_p"] = float(self.spec.get("top_p", 1.0))
        return kw

    def generate(self, messages, *, prefill=None, **overrides) -> str:
        import torch

        model, tokenizer, processor, kind = self._bundle()
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        messages = self._fold_system(messages)

        if kind == "causal":
            enc = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to(model.device)
            if prefill:
                tail = tokenizer(prefill, add_special_tokens=False, return_tensors="pt").to(model.device)
                enc["input_ids"] = torch.cat([enc["input_ids"], tail["input_ids"]], dim=-1)
                enc["attention_mask"] = torch.cat([enc["attention_mask"], tail["attention_mask"]], dim=-1)
            decode = tokenizer.decode
        else:
            chat = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
            enc = processor.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            ).to(model.device)
            decode = processor.decode

        enc.pop("token_type_ids", None)   # some templates emit it; generate() rejects it
        prompt_len = enc["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(**enc, **self._gen_kwargs(overrides))
        text = decode(out[0][prompt_len:], skip_special_tokens=True)
        return (prefill + text) if (prefill and kind == "causal") else text

    def perplexity(self, text: str) -> float:
        raise NotImplementedError("perplexity() lands in the defense Layer-1 milestone")


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
