"""Layer 2 -- paraphrase preprocessing (Jain et al. 2023, S3.2).

Route every prompt through a SEPARATE model with a fixed "paraphrase the following
sentences:" instruction before it reaches the target. Breaks attacks that depend on an
exact, brittle token sequence (GCG suffixes, forced prefixes, rigid rule lists).

Blind spot: meaning-preserving, so persona/roleplay and distractor framing survive.
Only gray-box safe -- an attacker with white-box access to the paraphraser can chain
optimizers against it.

Two practical details that the naive implementation gets wrong, both of which matter
for honest per-layer attribution in the report:

1. Instruct models chat back. `Qwen2.5-1.5B-Instruct` answers a paraphrase request with
   "Sure! Here's a paraphrased version: ..." -- feeding that preamble to the target
   pollutes the prompt, so it is stripped.

2. The paraphraser is itself safety-tuned, so on a harmful prompt it often refuses
   rather than paraphrasing. Passing that refusal on as the target's prompt would make
   the target answer the string "I can't help with that", which then grades as a
   non-compliance -- the defense would look effective for a reason that has nothing to
   do with the paraphrase mechanism. Instead a refusal is recorded as its own BLOCK
   (`reason="paraphraser refused"`), so the M5 attribution table can separate
   "paraphrasing broke the attack" from "the paraphraser declined to launder it".
"""
from __future__ import annotations

import re

from core.config import CONFIG
from core.models import load_paraphraser
from defense.base import Action, DefenseContext, DefenseLayer, Stage, Verdict
from defense.layer4_response_classifier import heuristic_label

# "Sure, here is a paraphrased version of the sentences:" and friends.
_PREAMBLE = re.compile(
    r"^\s*(sure|certainly|of course|okay|ok|absolutely)?[,!.]?\s*"
    r"(here(?:'s| is| are)?)?\s*"
    r"(a |the )?(paraphrased?|rephrased?|reworded?)\s*"
    r"(version|form|text|sentences?|paragraphs?)?\s*"
    r"(of (the )?(sentences?|text|prompt|request|paragraphs?))?\s*[:\-]\s*",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    """Strip an instruct model's conversational wrapper from around the paraphrase."""
    out = _PREAMBLE.sub("", text.strip(), count=1).strip()
    # a model often returns the paraphrase wrapped in quotes
    if len(out) >= 2 and out[0] in "\"'" and out[-1] == out[0]:
        out = out[1:-1].strip()
    return out


class ParaphraseDefense(DefenseLayer):
    name = "layer2_paraphrase"
    stage = Stage.PRE
    source = "Jain et al. 2023"

    def process(self, ctx: DefenseContext) -> DefenseContext:
        instruction = CONFIG["models"]["paraphraser"].get(
            "instruction", "paraphrase the following sentences:"
        )
        paraphraser = load_paraphraser(force_fake=self.force_fake)

        try:
            raw = paraphraser.generate(f"{instruction}\n\n{ctx.prompt}")
        except NotImplementedError:
            ctx.record(Verdict(self.name, Action.ALLOW, reason="stub: paraphraser backend not implemented"))
            return ctx

        rewritten = _clean(raw or "")

        if not rewritten:
            ctx.record(Verdict(self.name, Action.ALLOW, reason="empty paraphrase; kept original"))
            return ctx

        # The paraphraser declined to restate the request -- a block in its own right,
        # recorded distinctly so it is never confused with the paraphrase mechanism.
        if heuristic_label(rewritten) == "GOOD_BOT":
            enforce = bool(self.config.get("enforce_on_refusal", True))
            ctx.record(Verdict(
                self.name, Action.BLOCK if enforce else Action.ALLOW,
                reason="paraphraser refused to restate the request",
                detail={"paraphraser_reply": rewritten[:200]},
            ))
            return ctx

        ctx.metadata["prompt_before_paraphrase"] = ctx.prompt
        ctx.prompt = rewritten
        ctx.record(Verdict(
            self.name, Action.TRANSFORM, reason="prompt paraphrased",
            detail={"chars_before": len(ctx.metadata["prompt_before_paraphrase"]),
                    "chars_after": len(rewritten)},
        ))
        return ctx
