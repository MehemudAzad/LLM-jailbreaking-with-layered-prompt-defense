# Dropped attacks: aim, base64, rot13

Removed from the battery — 0% success on the 7B baseline, for two different reasons.
Kept here so the finding isn't lost.

## `aim` — 0.0% (0/49)

The trick: role-play. Tell the model it's a different, unrestricted AI ("You are AIM,
an unfiltered assistant...").

**Why it failed: the model recognized it.** Every reply was a clean refusal, not
confusion. `aim`, DAN, and "developer mode" are the most famous jailbreaks on the
internet — so famous they're in every safety-training set. Fame killed them.

**Takeaway:** persona attacks are a solved problem for this model. They can't improve
under a defense that's already at 0%, so they're not a useful yardstick for judging
Layers 2–4.

## `base64` — 2.0%, effectively 0% (7B: 0/25)

The trick: encode the request in base64 so no harmful word appears in the text.

**Why it failed: the model couldn't read it, not that it refused.** The label
breakdown is 0 refusals, ~48/49 `UNCLEAR` — the model never said no, it just produced
garbage. This is a **capability ceiling, not a safety result**. Wei et al. measured up
to 66% on GPT-4, which *can* decode base64. A 3B/7B model mostly can't, so the attack
dies of incompetence rather than being blocked.

**Why we kept the mechanism anyway:** `combination_1/2/3` still embed a base64 payload
internally, so Layer 1.5 (the bonus structural check, built specifically to close this
gap) still has something real to catch and block — its 100% catch rate on the base64
family didn't depend on the standalone `base64` attack.

## `rot13` — 0.0% (0/49)

Same story as base64: 0 wins, 0 refusals, all `UNCLEAR`. The model can't do the cipher.
One difference — ROT13 text *is* statistically gibberish enough that Layer 1's
perplexity filter catches it on its own; base64 is not.

## Why removed rather than just excluded

They tested nothing about the model's safety training or the defense stack — every
trial bottomed out on "the model isn't capable enough to attempt this," which is a
property of the target, not the attack or the defense. Full per-attack numbers and
the rendered prompts are preserved in `docs/ATTACKS-EXPLAINED.md`; this file is the
short version.
