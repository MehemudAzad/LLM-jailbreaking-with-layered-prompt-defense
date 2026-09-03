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

    def perplexity(self, text: str, *, window: int | None = None) -> float:
        """Perplexity of `text` under this model (exp of mean token NLL).

        `window=N` returns the worst (highest-perplexity) contiguous N-token window
        instead of the whole-sequence value -- so a short gibberish span can't hide
        inside a long fluent prompt. Used by defense Layer 1.
        """
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

    def perplexity(self, text: str, *, window: int | None = None) -> float:
        n = max(len(text.split()), 1)
        base = 50.0 + (int(hashlib.sha256(text.encode()).hexdigest(), 16) % 4000) / n
        return base * 1.6 if window else base


class TransformersModelHandle(ModelHandle):
    """Hugging Face Transformers backend (milestone M1).

    Loads the model once per (name, revision), cached on the class.
      - Plain text models: AutoModelForCausalLM + AutoTokenizer (e.g. Qwen2.5-3B-Instruct).
      - Text+vision models used text-only: falls back to AutoModelForImageTextToText +
        AutoProcessor (e.g. Gemma 3 4B) -- we never pass an image.

    Chat templates are applied as-is; if a template rejects a standalone `system` turn
    (Gemma does), the system text is folded into the next user turn and retried. Models
    with a real system slot (Qwen, Mistral, Llama) keep the role separation.

    All heavy imports are lazy, so `backend = "fake"` / `--dry-run` never touch torch.

    Optional model-spec keys (config.toml `[models.*]`):
      device   "cuda:0" / "cuda:1" -- pin to one GPU (default: device_map="auto")
      quant    "4bit"              -- bitsandbytes nf4 (e.g. the Qwen3.5-9B judge)
      thinking false               -- Qwen3/3.5: disable the reasoning block

    `generate()` = M1; `perplexity()` = M2; `quant`/`device`/`thinking` = M4 (judge).
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

    def _quant_config(self):
        """BitsAndBytesConfig when the model spec asks for `quant = "4bit"`, else None."""
        if str(self.spec.get("quant", "")).lower() not in ("4bit", "nf4", "int4"):
            return None
        import torch
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,   # T4 has no bf16
            bnb_4bit_use_double_quant=True,
        )

    def _bundle(self):
        key = (self.name, self._revision())
        cached = TransformersModelHandle._CACHE.get(key)
        if cached is not None:
            return cached

        import transformers as tf

        # transformers renamed `torch_dtype` -> `dtype` in 4.56
        _ver = tuple(int(x) for x in tf.__version__.split(".")[:2])
        _dtype_kw = "dtype" if _ver >= (4, 56) else "torch_dtype"

        device = self.spec.get("device")                       # e.g. "cuda:1" -- pin to one GPU
        load_kw = {"revision": self._revision(),
                   "device_map": ({"": device} if device else "auto")}
        qconf = self._quant_config()
        if qconf is not None:
            load_kw["quantization_config"] = qconf             # bnb owns the compute dtype
        else:
            load_kw[_dtype_kw] = self._resolve_dtype()

        try:
            tokenizer = tf.AutoTokenizer.from_pretrained(self.name, revision=self._revision())
            model = tf.AutoModelForCausalLM.from_pretrained(self.name, **load_kw)
            bundle = (model.eval(), tokenizer, "causal")
        except Exception:  # noqa: BLE001 -- not a plain CausalLM (Gemma 3 4B, Qwen3.5-9B, other VLMs)
            processor = tf.AutoProcessor.from_pretrained(self.name, revision=self._revision())
            model = tf.AutoModelForImageTextToText.from_pretrained(self.name, **load_kw)
            bundle = (model.eval(), processor, "vlm")

        TransformersModelHandle._CACHE[key] = bundle
        return bundle

    @staticmethod
    def _fold_system(messages: list[Message]) -> list[Message]:
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

    def _encode(self, tok, messages: list[Message], is_processor: bool):
        def shape(msgs):
            if is_processor:
                return [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in msgs]
            return msgs

        kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        if self.spec.get("thinking") is False:
            kw["enable_thinking"] = False        # Qwen3 / Qwen3.5 -- answer directly, no <think> block

        def apply(msgs):
            try:
                return tok.apply_chat_template(msgs, **kw)
            except TypeError:                    # this template doesn't accept enable_thinking
                kw.pop("enable_thinking", None)
                return tok.apply_chat_template(msgs, **kw)

        try:
            return apply(shape(messages))
        except Exception:  # noqa: BLE001 -- template rejects a standalone system turn (e.g. Gemma)
            return apply(shape(self._fold_system(messages)))

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

        model, tok, kind = self._bundle()
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        enc = self._encode(tok, messages, is_processor=(kind == "vlm")).to(model.device)

        if prefill and kind == "causal":
            tail = tok(prefill, add_special_tokens=False, return_tensors="pt").to(model.device)
            enc["input_ids"] = torch.cat([enc["input_ids"], tail["input_ids"]], dim=-1)
            enc["attention_mask"] = torch.cat([enc["attention_mask"], tail["attention_mask"]], dim=-1)

        enc.pop("token_type_ids", None)   # some templates emit it; generate() rejects it
        prompt_len = enc["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(**enc, **self._gen_kwargs(overrides))
        text = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        return (prefill + text) if (prefill and kind == "causal") else text

    def perplexity(self, text: str, *, window: int | None = None) -> float:
        import torch

        model, tok, _ = self._bundle()
        ctx_max = (getattr(model.config, "max_position_embeddings", None)
                   or getattr(model.config, "n_positions", 1024))
        ids = tok(text, return_tensors="pt", truncation=True, max_length=ctx_max)["input_ids"].to(model.device)
        if ids.shape[-1] < 2:
            return float("inf")

        with torch.no_grad():
            logits = model(ids).logits[0, :-1].float()          # (T-1, V): position t predicts token t+1
        nll = -torch.log_softmax(logits, dim=-1).gather(-1, ids[0, 1:, None]).squeeze(-1)   # (T-1,)

        if window is None or window >= nll.numel():
            return torch.exp(nll.mean()).item()

        csum = torch.cat([nll.new_zeros(1), nll.cumsum(0)])
        window_means = (csum[window:] - csum[:-window]) / window                            # (T-1-window,)
        return torch.exp(window_means.max()).item()


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
