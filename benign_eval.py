#!/usr/bin/env python3
"""Cost-of-defense measurement: what the stack does to *ordinary* requests.

Blocking harmful prompts is only half the design question. A filter that also refuses
homework help is not a working defense, it is a broken assistant. M3 already pinned
Layer 1's threshold for 0% false positives on the frozen benign set; this measures the
same thing for the *whole assembled stack*, where L2's paraphrase and L3's refusal
priming can degrade a benign answer without any layer "blocking" anything.

Run both arms, then compare:

    import benign_eval
    und = benign_eval.run(defended=False, tag='m5benign-und')
    dfn = benign_eval.run(defended=True,  tag='m5benign-def')
    benign_eval.print_compare(und, dfn)

Metrics per prompt:
  blocked / blocked_by  a PRE layer stopped a harmless request -- a false positive
  refused               the model produced a refusal (Layer 4's refusal heuristic)
  served                neither blocked nor refused -- the user got a real answer
  chars                 reply length, a crude proxy for "did the answer get worse"

`served_%` is the headline: undefended vs defended, on identical prompts and seed.
Deliberately a repo module, not notebook cells -- notebook cells do not sync from git,
imported modules do.
"""
from __future__ import annotations

import json
import pathlib
import time

import pandas as pd

from core.config import CONFIG
from core.datasets import load_benign
from core.models import load_target
from core.seed import seed_everything
from core.transcript import TranscriptLogger
from defense import build_pipeline
from defense.base import DefenseContext
from defense.layer4_response_classifier import heuristic_label


def run(defended: bool, limit: int | None = None, force_fake: bool = False,
        tag: str = "", progress: bool = True) -> pathlib.Path:
    """Send every benign prompt through the target, with or without the stack."""
    seed_everything()
    prompts = load_benign()
    if limit:
        prompts = prompts[:limit]

    target = load_target(force_fake=force_fake)
    pipeline = build_pipeline(force_fake=force_fake) if defended else None

    tag = tag or ("m5benign-def" if defended else "m5benign-und")
    t0 = time.time()
    with TranscriptLogger(tag=tag) as log:
        log.log({"type": "config", "defended": defended, "n_prompts": len(prompts),
                 "target": CONFIG["models"]["target"]["name"]})

        for i, bp in enumerate(prompts, 1):
            ctx = DefenseContext(goal_id=bp.id, attack="benign", original_prompt=bp.text,
                                 prompt=bp.text)
            if pipeline is not None:
                ctx = pipeline.run_pre(ctx)

            if not ctx.blocked:
                messages = []
                if ctx.system:
                    messages.append({"role": "system", "content": ctx.system})
                messages.append({"role": "user", "content": ctx.prompt})
                ctx.response = target.generate(messages)
                if pipeline is not None:
                    ctx = pipeline.run_post(ctx)

            refused = bool(ctx.response) and heuristic_label(ctx.response) == "GOOD_BOT"
            log.log({
                "type": "benign_trial",
                "id": bp.id,
                "defended": defended,
                "prompt": bp.text,
                "prompt_sent": ctx.prompt,
                "response": ctx.response,
                "blocked": ctx.blocked,
                "blocked_by": ctx.blocked_by,
                "refused": refused,
                "served": (not ctx.blocked) and not refused,
                "chars": len(ctx.response or ""),
                "verdicts": [v.__dict__ for v in ctx.verdicts],
            })
            if progress:
                state = f"blocked:{ctx.blocked_by}" if ctx.blocked else ("refused" if refused else "served")
                eta = (time.time() - t0) / i * (len(prompts) - i) / 60
                print(f"  [{i:3}/{len(prompts)}] {bp.id:<8} {state:<28} eta {eta:4.1f}m", flush=True)

        run_dir = log.dir
    print(f"\n{'defended' if defended else 'undefended'} benign pass -> {run_dir}")
    return run_dir


def load(run_dir: str | pathlib.Path) -> pd.DataFrame:
    path = pathlib.Path(run_dir) / "transcript.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return pd.DataFrame([r for r in rows if r.get("type") == "benign_trial"])


def summarise(run_dir: str | pathlib.Path) -> dict:
    df = load(run_dir)
    n = len(df)
    return {
        "n": n,
        "served_%": round(100 * df.served.sum() / n, 1) if n else None,
        "refused_%": round(100 * df.refused.sum() / n, 1) if n else None,
        "blocked_%": round(100 * df.blocked.sum() / n, 1) if n else None,
        "median_chars": int(df.chars.median()) if n else 0,
    }


def print_compare(undefended_dir, defended_dir) -> pd.DataFrame:
    """The cost-of-defense table: identical benign prompts, stack off vs stack on."""
    und, dfn = summarise(undefended_dir), summarise(defended_dir)
    tbl = pd.DataFrame([und, dfn], index=["undefended", "defended"])
    tbl.loc["delta"] = tbl.loc["defended"] - tbl.loc["undefended"]

    print("=== COST OF DEFENSE - 50 benign prompts, identical seed ===")
    print(f"\n{tbl.to_string()}")

    d = load(defended_dir)
    fp = d[d.blocked]
    print(f"\nbenign false positives (a PRE layer blocked a harmless request): {len(fp)}/{len(d)}")
    if len(fp):
        print(fp.blocked_by.value_counts().to_string())
        for _, r in fp.head(5).iterrows():
            print(f"   {r.id}  blocked_by={r.blocked_by}  {r.prompt[:90]!r}")

    newly_refused = d[d.refused & ~d.blocked]
    print(f"\nrefused (not blocked, model itself declined): {len(newly_refused)}/{len(d)}")
    for _, r in newly_refused.head(5).iterrows():
        print(f"   {r.id}  {r.prompt[:90]!r}")
    return tbl


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--defended", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run(defended=a.defended, limit=a.limit, force_fake=a.dry_run)
