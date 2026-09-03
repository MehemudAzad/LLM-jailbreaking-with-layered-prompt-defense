"""LayeredDefense -- chains the layers around the target model.

    ctx = pipeline.run_pre(ctx)          # PRE layers; short-circuits on the first BLOCK
    if not ctx.blocked:
        ctx.response = target.generate(...)
        ctx = pipeline.run_post(ctx)      # POST layers
"""
from __future__ import annotations

from defense.base import DefenseContext, DefenseLayer, Stage


class LayeredDefense:
    def __init__(self, layers: list[DefenseLayer]):
        self.layers = layers
        self.pre = [l for l in layers if l.stage == Stage.PRE and l.enabled]
        self.post = [l for l in layers if l.stage == Stage.POST and l.enabled]

    def run_pre(self, ctx: DefenseContext) -> DefenseContext:
        for layer in self.pre:
            ctx = layer.process(ctx)
            if ctx.blocked:
                break
        return ctx

    def run_post(self, ctx: DefenseContext) -> DefenseContext:
        for layer in self.post:
            ctx = layer.process(ctx)
        return ctx

    def describe(self) -> list[str]:
        return [f"{l.stage.value}:{l.name}" for l in self.pre + self.post]
