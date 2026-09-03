"""adaptive -- run the whole battery per goal; count a success if ANY member succeeds.

This is the project's actual "battery" strategy (Wei et al. 2023 report 96-100% with it).
It's an orchestration strategy, not a prompt wrapper -- run_eval.py handles the loop and
skips this entry in the per-technique sweep.
"""
from attacks import register
from attacks.base import AttackOutput, AttackTechnique


@register
class Adaptive(AttackTechnique):
    name = "adaptive"
    category = "combination"
    source = "Wei et al. 2023 (section 5)"

    def members(self) -> list[AttackTechnique]:
        """Every other registered technique (call from run_eval.py's battery loop)."""
        from attacks import load_all

        return [t for name, t in load_all().items() if name not in ("adaptive", "passthrough")]

    def apply(self, goal, *, helper=None) -> AttackOutput:
        # Single-shot callers get the strongest member; the real battery loop lives in run_eval.py.
        from attacks.combination import Combination3

        return Combination3().apply(goal, helper=helper)
