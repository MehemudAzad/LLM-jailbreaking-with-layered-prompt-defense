"""Layer 1 -- perplexity filter (plain + windowed).

Jain et al. 2023 (S3.1); Alon & Kamfonas 2023. Score each incoming prompt's perplexity
under a frozen scorer LM (`models.perplexity_scorer`, default gpt2-large). Reject it if
the score exceeds a threshold calibrated so that no clean reference prompt is flagged
(Jain et al.'s method -- see notebooks/m2_perplexity_filter.ipynb).

- plain    : perplexity of the whole prompt.
- windowed : the worst (highest-perplexity) contiguous `window_size`-token span, so a
             short adversarial suffix can't be diluted by a long fluent prefix. This is
             what the BLOCK decision uses when `windowed = true`.

Blind spot: a gibberish detector, not a jailbreak detector. Fluent human-authored
attacks (AIM, roleplay, prefix injection) score like normal English and pass straight
through -- that's expected, and why Layers 2-4 exist.

Config (`[defense.layer1_perplexity]`): `threshold` (0 = never block, pre-calibration),
`window_size`, `windowed`, `enforce`.
"""
from __future__ import annotations

from core.models import load_perplexity_scorer
from defense.base import Action, DefenseContext, DefenseLayer, Stage, Verdict


class PerplexityFilter(DefenseLayer):
    name = "layer1_perplexity"
    stage = Stage.PRE
    source = "Jain et al. 2023; Alon & Kamfonas 2023"

    def process(self, ctx: DefenseContext) -> DefenseContext:
        threshold = float(self.config.get("threshold", 0.0))
        window = int(self.config.get("window_size", 16))
        windowed = bool(self.config.get("windowed", True))
        enforce = bool(self.config.get("enforce", False))

        scorer = load_perplexity_scorer(force_fake=self.force_fake)
        ppl = scorer.perplexity(ctx.prompt)
        win_ppl = scorer.perplexity(ctx.prompt, window=window) if windowed else ppl
        decisive = win_ppl if windowed else ppl

        over = threshold > 0.0 and decisive > threshold
        action = Action.BLOCK if (over and enforce) else Action.ALLOW
        ctx.record(Verdict(
            self.name, action, score=decisive,
            reason=(f"ppl={ppl:.0f} windowed_ppl={win_ppl:.0f} "
                    f"threshold={threshold:.0f} over={over} enforce={enforce}"),
            detail={"perplexity": ppl, "windowed_perplexity": win_ppl, "window_size": window},
        ))
        return ctx
