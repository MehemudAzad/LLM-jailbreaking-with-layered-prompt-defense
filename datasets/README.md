# Datasets — frozen for the whole project

Two fixed sets, reused for every run. Counts are pinned in `config.toml` `[datasets]`.

## `harmful_behaviors.jsonl` — **git-ignored**, build it

```bash
python datasets/build_harmful.py          # 50 rows (default), seed 1337
python datasets/build_harmful.py 520      # the full benchmark
```

Deterministic: same seed → same selection. The script pulls from, in order:

1. `datasets/advBench/**/*.parquet` — the [`walledai/AdvBench`](https://huggingface.co/datasets/walledai/AdvBench)
   mirror if you've downloaded it (needs `pandas` + `pyarrow`)
2. `raw.githubusercontent.com/llm-attacks/llm-attacks/.../harmful_behaviors.csv` — the
   original (Zou et al. 2023), ungated, stdlib-only

Both are the same 520 `(goal, target)` rows, MIT-licensed. The frozen file is git-ignored
because the repo is public; anyone regenerates the exact set with the command above.

Row shape:

    {"id": "hb_0001", "category": "advbench", "goal": "<harmful instruction>",
     "target_prefix": "Sure, here is a", "source": "AdvBench harmful_behaviors #12 (Zou et al. 2023)"}

- `goal` — the instruction an attack tries to get the target to follow.
- `target_prefix` — first clause of AdvBench's `target`; the affirmative opener for
  `prefix_injection` / `combination_*`.

## `benign_prompts.jsonl` — committed

50 harmless instructions written to **mirror AdvBench's imperative style** ("Write a
script that…", "Give step-by-step instructions for…") so the Layer 1 false-positive rate
is measured on a fair control group. Also used to measure the benign-quality drop the
defense stack costs.

    {"id": "bn_0001", "text": "<an ordinary imperative request>"}

## `advBench/` — the raw download

Kept for provenance (`README.md` only is tracked; the parquet is git-ignored). Delete it
and the builder falls back to the GitHub CSV.

## What goes in git

`build_harmful.py`, `benign_prompts.jsonl`, `advBench/README.md`. Not the frozen harmful
set, not the parquet. Model outputs go to `logs/` — see `logs/README.md`.
