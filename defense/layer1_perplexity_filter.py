"""Layer 1 -- perplexity filter (plain + windowed).

Jain et al. 2023 (S3.1); Alon & Kamfonas 2023. Reject a prompt whose log-perplexity
under a frozen scorer LM exceeds a threshold calibrated on our own benign set. The
windowed variant also scores contiguous token windows, so a locally-gibberish
adversarial suffix can't hide inside an otherwise fluent prompt.

Blind spot: a gibberish detector, not a jailbreak detector. Fluent human-authored
attacks (AIM, roleplay, prefix injection) sail through -- their token statistics look
perfectly normal.

STATUS: scores + logs, does NOT block yet (`enforce = false` in config.toml).
TODO(Member B):
  * real scoring via core.models.load_perplexity_scorer().perplexity()
  * calibrate `threshold` on datasets/benign_prompts.jsonl (target FPR ~6-10%)
  * implement the windowed minimum when config `windowed = true`
  * flip `enforce = true` once calibrated
"""
from __future__ import annotations

from core.models import load_perplexity_scorer
from defense.base import Action, DefenseContext, DefenseLayer, Stage, Verdict


class PerplexityFilter(DefenseLayer):
    name = "layer1_perplexity"
    stage = Stage.PRE
    source = "Jain et al. 2023; Alon & Kamfonas 2023"

    def process(self, ctx: DefenseContext) -> DefenseContext:
        threshold = float(self.config.get("threshold", 1000.0))
        enforce = bool(self.config.get("enforce", False))
        scorer = load_perplexity_scorer(force_fake=self.force_fake)

        try:
            score = scorer.perplexity(ctx.prompt)
        except NotImplementedError:
            ctx.record(Verdict(self.name, Action.ALLOW, reason="stub: scorer backend not implemented"))
            return ctx

        over = score > threshold
        action = Action.BLOCK if (over and enforce) else Action.ALLOW
        ctx.record(Verdict(
            self.name, action, score=score,
            reason=f"log-ppl={score:.1f} threshold={threshold:.1f} over={over} enforce={enforce}",
        ))
        return ctx
