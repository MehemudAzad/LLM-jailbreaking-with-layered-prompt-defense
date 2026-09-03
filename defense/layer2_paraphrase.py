"""Layer 2 -- paraphrase preprocessing (Jain et al. 2023, S3.2).

Route every prompt through a SEPARATE model with a fixed "paraphrase the following
sentences:" instruction before it reaches the target. Breaks attacks that depend on an
exact, brittle token sequence (GCG suffixes, base64 payloads, forced prefixes).

Blind spot: meaning-preserving, so persona/roleplay and distractor framing survive.
Only gray-box safe -- an attacker with white-box access to the paraphraser can chain
optimizers against it.

STATUS: pass-through until the paraphraser backend exists.
TODO(Member B): on success, stash the original in ctx.metadata and replace ctx.prompt.
"""
from __future__ import annotations

from core.config import CONFIG
from core.models import load_paraphraser
from defense.base import Action, DefenseContext, DefenseLayer, Stage, Verdict


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
            rewritten = paraphraser.generate(f"{instruction}\n\n{ctx.prompt}").strip()
        except NotImplementedError:
            ctx.record(Verdict(self.name, Action.ALLOW, reason="stub: paraphraser backend not implemented"))
            return ctx

        if rewritten:
            ctx.metadata["prompt_before_paraphrase"] = ctx.prompt
            ctx.prompt = rewritten
            ctx.record(Verdict(self.name, Action.TRANSFORM, reason="prompt paraphrased"))
        else:
            ctx.record(Verdict(self.name, Action.ALLOW, reason="empty paraphrase; kept original"))
        return ctx
