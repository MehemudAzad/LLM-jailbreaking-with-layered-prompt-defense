#!/usr/bin/env python3
"""Evaluation harness -- the one place the attack battery and the defense stack meet.

    python run_eval.py --dry-run                  # fake models everywhere; walk it end to end
    python run_eval.py --attack prefix_injection  # one technique, real models
    python run_eval.py --attack all --defense on  # the full evaluation pass
    python run_eval.py --attack all --defense off # baseline ASR (undefended target, judge still grades)

Every (goal x attack) trial writes exactly one JSONL record to
`logs/<run-id>/transcript.jsonl`. Those files are the evidence base for both
reports and the demo -- never edit one after the fact, re-run instead.

FROZEN after Week 10 at the interface level. The body grows as both halves land.
"""
from __future__ import annotations

import argparse
import sys
import time

from core.config import CONFIG
from core.seed import seed_everything
from core.transcript import TranscriptLogger
from core.datasets import load_harmful, Goal
from core.models import load_target, load_helper
from attacks import load_all as load_attacks
from defense import build_pipeline
from defense.base import DefenseContext


def run_trial(goal: Goal, attack, target, helper, pipeline, use_defense: bool, grader=None) -> DefenseContext:
    """One goal through one attack, optionally through the defense stack.

    `grader` (a ResponseClassifier) labels the reply when the defense stack is off, so a
    baseline run still gets BAD_BOT/GOOD_BOT/UNCLEAR without blocking anything.
    """
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
        elif grader is not None:
            ctx = grader.process(ctx)

    return ctx


def _synthetic_goals() -> list[Goal]:
    return [Goal(id="dryrun-0001", goal="<<synthetic dry-run goal>>", category="placeholder")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack", default="all", help="technique name (see attacks/), or 'all'")
    ap.add_argument("--defense", choices=["on", "off"], default="on")
    ap.add_argument("--grade", dest="grade", action="store_true", default=True,
                    help="label each reply with the judge (default; only matters when --defense off)")
    ap.add_argument("--no-grade", dest="grade", action="store_false")
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

    grader = None
    if args.grade and not use_defense:
        from defense.layer4_response_classifier import ResponseClassifier
        grader = ResponseClassifier({"enabled": True, "enforce": False, "_force_fake": force_fake})

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
    by_attack: dict[str, dict[str, int]] = {}
    run_names = [n for n in sorted(attacks) if n != "adaptive"]
    total = len(goals) * len(run_names)
    t_start = time.time()

    with TranscriptLogger(tag=tag) as log:
        log.log({
            "type": "config", "seed": seed, "attack": args.attack, "defense": use_defense,
            "grade": bool(grader) or use_defense, "dry_run": args.dry_run,
            "n_goals": len(goals), "n_attacks": len(run_names), "attacks": run_names,
        })
        print(f"{total} trials ({len(goals)} goals x {len(run_names)} attacks)  "
              f"defense={'on' if use_defense else 'off'}  grade={bool(grader) or use_defense}", flush=True)
        for goal in goals:
            for name in run_names:
                attack = attacks[name]
                ctx = run_trial(goal, attack, target, helper, pipeline, use_defense, grader)
                label = ctx.metadata.get("response_label", "N/A")
                labels[label] = labels.get(label, 0) + 1
                by_attack.setdefault(name, {})[label] = by_attack.setdefault(name, {}).get(label, 0) + 1
                trials += 1
                rate = trials / max(time.time() - t_start, 1e-6)
                eta = (total - trials) / rate / 60
                tail = f"  blocked:{ctx.blocked_by}" if ctx.blocked else ""
                print(f"  [{trials:>4}/{total}] {goal.id:<9} {name:<22} {label:<9}"
                      f"  eta {eta:4.1f}m{tail}", flush=True)
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
                    "label_method": ctx.metadata.get("label_method"),
                    "verdicts": [vars(v) for v in ctx.verdicts],
                    "metadata": ctx.metadata,
                })

        def _asr(counts: dict[str, int]) -> float | None:
            n = sum(counts.values())
            return round(counts.get("BAD_BOT", 0) / n, 4) if n else None

        summary = {
            "type": "summary", "trials": trials, "labels": labels,
            "asr": _asr(labels),
            "asr_by_attack": {n: _asr(c) for n, c in sorted(by_attack.items())},
            "by_attack": by_attack,
            "run_dir": str(log.dir),
        }
        log.log(summary)

    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min -> {log.path}")
    print(f"overall ASR {summary['asr']}  labels {labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
