#!/usr/bin/env python3
"""Evaluation harness -- the one place the attack battery and the defense stack meet.

    python run_eval.py --dry-run                  # fake models everywhere; walk it end to end
    python run_eval.py --attack prefix_injection  # one technique, real models
    python run_eval.py --attack all --defense on  # the full evaluation pass
    python run_eval.py --attack all --defense off # baseline ASR (undefended target)

Every (goal x attack) trial writes exactly one JSONL record to
`logs/<run-id>/transcript.jsonl`. Those files are the evidence base for both
reports and the demo -- never edit one after the fact, re-run instead.

FROZEN after Week 10 at the interface level. The body grows as both halves land.
"""
from __future__ import annotations

import argparse
import sys

from core.config import CONFIG
from core.seed import seed_everything
from core.transcript import TranscriptLogger
from core.datasets import load_harmful, Goal
from core.models import load_target, load_helper
from attacks import load_all as load_attacks
from defense import build_pipeline
from defense.base import DefenseContext


def run_trial(goal: Goal, attack, target, helper, pipeline, use_defense: bool) -> DefenseContext:
    """One goal through one attack, optionally through the defense stack."""
    out = attack.apply(goal.goal, helper=helper if attack.uses_helper_lm else None)

    ctx = DefenseContext(
        goal_id=goal.id,
        attack=attack.name,
        original_prompt=goal.goal,
        prompt=out.prompt,
        system=out.system or "",
        prefill=out.prefill,
        metadata={"goal": goal.goal, "decode": out.decode, **out.metadata},
    )

    if use_defense:
        ctx = pipeline.run_pre(ctx)

    if not ctx.blocked:
        messages = []
        if ctx.system:
            messages.append({"role": "system", "content": ctx.system})
        messages.append({"role": "user", "content": ctx.prompt})
        ctx.response = target.generate(messages, prefill=ctx.prefill)
        if use_defense:
            ctx = pipeline.run_post(ctx)

    return ctx


def _synthetic_goals() -> list[Goal]:
    return [Goal(id="dryrun-0001", goal="<<synthetic dry-run goal>>", category="placeholder")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack", default="all", help="technique name (see attacks/), or 'all'")
    ap.add_argument("--defense", choices=["on", "off"], default="on")
    ap.add_argument("--dry-run", action="store_true", help="force backend=fake for every model")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of goals")
    ap.add_argument("--tag", default="", help="label folded into the run-id")
    args = ap.parse_args(argv)

    seed = seed_everything()
    force_fake = args.dry_run

    attacks = load_attacks()
    if args.attack != "all":
        if args.attack not in attacks:
            print(f"unknown attack {args.attack!r}. known: {', '.join(sorted(attacks))}", file=sys.stderr)
            return 2
        attacks = {args.attack: attacks[args.attack]}

    target = load_target(force_fake=force_fake)
    helper = load_helper(force_fake=force_fake)
    pipeline = build_pipeline(force_fake=force_fake)
    use_defense = args.defense == "on"

    try:
        goals = load_harmful()
    except FileNotFoundError:
        print("datasets/harmful_behaviors.jsonl not found -- see datasets/README.md", file=sys.stderr)
        return 1

    real_goals = [g for g in goals if not g.goal.startswith("<<")]
    if not real_goals:
        if not args.dry_run:
            print("harmful set still holds only schema placeholders -- populate it "
                  "(datasets/README.md) before a real run.", file=sys.stderr)
            return 1
        goals = _synthetic_goals()
    else:
        goals = real_goals
    if args.limit:
        goals = goals[: args.limit]

    tag = args.tag or ("dryrun" if args.dry_run else ("defended" if use_defense else "baseline"))
    trials = 0
    labels: dict[str, int] = {}

    with TranscriptLogger(tag=tag) as log:
        log.log({
            "type": "config", "seed": seed, "attack": args.attack, "defense": use_defense,
            "dry_run": args.dry_run, "n_goals": len(goals), "n_attacks": len(attacks),
            "attacks": sorted(attacks),
        })
        for goal in goals:
            for name, attack in attacks.items():
                if name == "adaptive":
                    # TODO(joint): battery loop -- run every other member, success if ANY is BAD_BOT.
                    continue
                ctx = run_trial(goal, attack, target, helper, pipeline, use_defense)
                label = ctx.metadata.get("response_label", "N/A")
                labels[label] = labels.get(label, 0) + 1
                trials += 1
                log.log({
                    "type": "trial",
                    "goal_id": goal.id,
                    "attack": name,
                    "category": attack.category,
                    "defense": use_defense,
                    "blocked": ctx.blocked,
                    "blocked_by": ctx.blocked_by,
                    "prompt_sent": ctx.prompt,
                    "system": ctx.system,
                    "response": ctx.response,
                    "label": label,
                    "verdicts": [vars(v) for v in ctx.verdicts],
                    "metadata": ctx.metadata,
                })

        summary = {
            "type": "summary", "trials": trials, "labels": labels,
            "asr": round(labels.get("BAD_BOT", 0) / trials, 4) if trials else None,
            "run_dir": str(log.dir),
        }
        log.log(summary)

    print(f"done -> {log.path}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
