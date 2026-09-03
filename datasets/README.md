# Datasets — frozen for the whole project

Two small, fixed sets, reused for every run from Sprint 1 to week 13. Freeze them once
(~15–20 each per the sprint plan), record the counts in `config.toml`
(`[datasets] harmful_count` / `benign_count`), then don't touch them again.

## `harmful_behaviors.jsonl`

One JSON object per line:

    {"id": "hb_0001", "category": "...", "goal": "<a single harmful instruction>",
     "target_prefix": "Sure, here is", "source": "AdvBench"}

- `goal` — one imperative instruction; the thing an attack tries to get the target to do.
- `target_prefix` — the affirmative opener used by `prefix_injection` / `combination_*`.
- `source` — where the row came from, for the report's reproducibility section.

**Populate from a published benchmark; don't write these yourself.** The standard choice
is **AdvBench** (`harmful_behaviors.csv`) from Zou et al. 2023, *Universal and
Transferable Adversarial Attacks on Aligned Language Models* —
<https://github.com/llm-attacks/llm-attacks>. Wei et al. 2023 also publish a curated set
of 32 behaviours. Pick ~15–20, keep the selection fixed, cite it.

The rows in the file now are **schema placeholders** (their `goal` starts with `<<`);
`run_eval.py` refuses a real (non `--dry-run`) run until they're replaced.

## `benign_prompts.jsonl`

    {"id": "bn_0001", "text": "<an ordinary request>"}

The control group: same size and rough topic spread as the harmful set. Used to
calibrate the Layer 1 perplexity threshold (target false-positive rate ~6–10%) and to
measure the benign-quality drop the defense stack costs.

## What goes in git

These two `.jsonl` files, yes. The model outputs they produce go to `logs/` — see
`logs/README.md`.
