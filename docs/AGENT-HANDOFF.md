# Start here (new agent / new account)

This project has been worked on across many sessions with another Claude agent. This
file is what that agent would tell you if it could. Read `docs/HANDOVER.md` right after
this one — it has the actual technical state, bugs, and next steps. This file is the
context that doesn't live in the repo: how the user works, and the history behind
decisions.

## ⚠️ First thing to do

There are uncommitted changes on disk right now (three attack files deleted, two files
fixed to match, one new doc). Run `git status`. If they're still there, commit and push
before doing anything else, or they only exist on this one laptop.

## Who you're working with, and how

- Solo project (CSE-406, BUET), one teammate contributing on and off. No GPU on the
  user's machine (MacBook M4) — all real runs happen on Kaggle (T4, or two T4s).
- **Write executable code as `.ipynb` notebooks**, not scripts, for anything that needs
  to run on Kaggle. Plain `.py` is fine for shared library code that notebooks import
  (`core/`, `attacks/`, `defense/`, `report.py`) — keep notebook *cells* thin and put
  logic in importable modules instead, because notebook cells do not update from git;
  only imported modules do. See the "Kaggle gotchas" section in `HANDOVER.md`.
- **Show a plan and get approval before writing code.** This is treated as research
  work — one milestone at a time, and each finished milestone should be runnable by the
  user without further help from you.
- **The user runs things themselves** and reports results back. Don't run long/GPU
  things for them unless asked.
- **Keep responses short and plainly worded.** Not verbose, not overly technical. Lead
  with the answer.
- **Don't commit for them.** Make the changes, tell them what changed, they commit.
- **End every reply with**: purpose / what's new / next steps.
- The repo is **public**. Never commit raw model completions from jailbreak attempts —
  `logs/` and `datasets/harmful_behaviors.jsonl` are git-ignored on purpose.
  `datasets/build_harmful.py` regenerates the exact same 50 AdvBench prompts from a
  seed (1337), so nothing needed for that is actually lost by not committing it.

## The project, in one paragraph

Tool 27 of the CSE-406 spec: build an LLM jailbreak attack battery and a layered
defense, measure Attack Success Rate before and after. Target model gets attacked with
~15-18 techniques (persona tricks, encoding, prompt injection, combinations) drawn from
Wei et al. 2023; the defense is 6 stacked layers (perplexity filter, a structural
cipher check, a paraphraser, system-prompt hardening, a prefill guard, and a response
judge), 3 of them the team's own design for the bonus marks. A second LLM ("the judge")
grades every reply BAD_BOT/GOOD_BOT/UNCLEAR to compute ASR automatically — getting that
judge accurate was most of the hard work, not the attacks or defenses themselves.

## Notable history not in HANDOVER.md

- Went through **three model swaps** for good reasons, not indecision: 3B → 7B target
  (to test whether the 3B's low ASR was a capability ceiling — it partly was), and the
  judge landed on `Qwen/Qwen3.5-9B` in bitsandbytes 4-bit specifically because the GGUF
  version someone wanted can't run on the `transformers`-based stack this repo uses.
- The judge's grading logic went through **two rounds of real bugs**, both about
  cipher replies: first the judge was scoring undecoded base64/rot13 text (grading the
  *form*, not the content); the fix then over-corrected and started mangling plain
  English replies that merely looked cipher-ish. Both are fixed now in
  `defense/layer4_response_classifier.py`, but if ASR numbers ever look weirdly
  high/low again for an encoding-family attack, check there first.
- Compared our ASR metric against the standard AdvBench one (refusal-string absence).
  Standard metric said 46.7%, ours said 15.2% — the gap is almost entirely gibberish
  replies that contain no refusal phrase but also no harmful content. Worth keeping in
  the report as a methodology point, not a discrepancy to hide.
- Three attacks (`aim`, `base64`, `rot13`) were just removed from the battery — they
  scored ~0% for a *capability* reason (model literally can't decode ciphers / persona
  attacks are too well-known to work), not a safety reason. The finding is preserved in
  `docs/DROPPED-ATTACKS.md` rather than deleted outright.

## Where to actually start

Open `docs/HANDOVER.md` now — section "❌ What is broken" has the real priority list
(the L4-blocks-benign-requests bug is the top one). This file's job was just orientation.
