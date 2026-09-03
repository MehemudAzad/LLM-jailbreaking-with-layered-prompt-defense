# Sprint 1 — Getting Started (Now → end of Week 10)

Goal by the end of this sprint: one command that takes a harmful prompt, runs it
(optionally through an attack), passes it through a defense pipeline (stubbed for now),
hits your target model, and logs the response. Prove the shape works end to end before
going deep on any one piece.

## 1. Pin your target model

Use **`Qwen/Qwen2.5-3B-Instruct`** as the primary, pinned target for all graded runs:

- text-only (`Qwen2ForCausalLM`) — no multimodal / image handling to deal with
- ungated on Hugging Face — no licence gate, no token needed to pull the weights
- genuinely safety-tuned, so jailbreak attempts against it mean something
- 3B — small enough to run the full battery repeatedly on a Kaggle T4 without long waits
- has a real `system` prompt slot, which defense Layer 3 (system hardening) relies on

Secondary targets for the optional week-12 cross-check (to show the stack isn't specific
to one model): **`google/gemma-3-4b-it`** or **`mistralai/Mistral-7B-Instruct-v0.3`**.
Optional polish, not a requirement.

Naming note: `Qwen2` / `Qwen2.5-Instruct` are text-only models; the vision line is a
*separate* family, **Qwen2-VL / Qwen2.5-VL**. We use the plain text `Qwen2.5-3B-Instruct`.
(This section previously said "skip Qwen2, it's a vision-language model" — that was wrong.)

Pin it now — this is the one decision everything else depends on. It lives in
`config.toml` under `[models.target]`; `notebooks/m1_model_backend.ipynb` prints the
current Hub commit hash to paste into `revision` (replacing `PIN-ME`) before any
non-`--dry-run` run.

## 2. Set up the shared project skeleton

Create one repo with a config file holding the pinned model name, the
paraphraser/scorer model name, and your random seed. Add three folders:

- `attacks/` — one file per technique
- `defense/` — one file per layer
- `logs/` — raw transcripts

This is the scaffolding both of you build into in parallel without stepping on each
other.

## 3. Write the one shared function first, together

Before splitting up, pair on a single function:

```
send_prompt(text) -> response
```

that calls your pinned model and returns its output. Every attack wraps a prompt
before this call; every defense layer wraps around this call too. Getting this right
together now avoids two incompatible versions later.

## 4. Assemble a small prompt set

Write (or gather) ~15–20 short harmful-behavior instructions for testing attacks, and
~15–20 ordinary benign instructions as your control group for measuring false
positives and quality drop later. Keep this set small and fixed — you'll reuse it for
every run from here to week 13.

## 5. Implement your first 2–3 attacks to prove the loop

Don't start with all 8–10 techniques. Implement `prefix_injection`,
`refusal_suppression`, and `base64` first — they're the simplest to code and cover
both failure modes (competing objectives and mismatched generalization). Run them
against `send_prompt()` and confirm you get back real, logged responses.

## 6. Stub the four defense layers as pass-through

Write `perplexity_filter()`, `paraphrase()`, `harden()`, and `output_check()` as
functions that currently just return their input unchanged. This isn't wasted work —
it means the full attacker → defense → model → response shape exists end to end
today, and either of you can fill in a real layer later without blocking the other
person's progress.

---

**Suggested split:** do steps 1–3 together, since those decisions affect both of you.
Then one person runs with steps 4–5 (attack side) while the other does step 6 and
starts reading up on how to calibrate the perplexity threshold for next week (defense
side).
