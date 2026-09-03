"""Control condition: send the goal unmodified. Establishes baseline ASR."""
from attacks import register
from attacks.base import AttackOutput, AttackTechnique


@register
class Passthrough(AttackTechnique):
    name = "passthrough"
    category = "control"
    source = "baseline / control condition"

    def apply(self, goal, *, helper=None):
        return AttackOutput(prompt=goal)
