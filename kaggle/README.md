# Pushing notebooks to Kaggle from the terminal

Lets you edit `notebooks/*.ipynb` locally and run them on Kaggle's GPUs without
opening kaggle.com, using the official `kaggle` CLI. This does **not** give you
a live/interactive kernel (VS Code's Jupyter extension has nothing on Kaggle to
attach to) — it's push-and-poll: push, wait, pull the result.

Both kernels here are already wired to `khalidhasantuhin` (`kernel-metadata.json`'s
`id` field) — nothing left to fill in there.

## One-time setup

1. Install the CLI (already done on this machine): `pip3 install --user kaggle`
   — its entry point landed outside `PATH`, so every command below uses
   `python3 -m kaggle` instead of the bare `kaggle` command. `kaggle/run.sh`
   already does this for you.
2. Authenticate — either works:
   - **OAuth (recommended, no token file to manage):**
     ```
     python3 -m kaggle auth login
     ```

     Opens a browser flow; credentials are cached for you.
   - **API token:**
     Go to [https://www.kaggle.com/settings/api](https://www.kaggle.com/settings/api) → "Generate New Token", then either
     ```
     export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx     # put this in your shell profile
     ```

     or save the token string to `~/.kaggle/access_token`.
3. **Not required, but optional if you hit rate limits:** both notebooks look for
   `GH_TOKEN` and `HF_TOKEN` via `kaggle_secrets.UserSecretsClient` and print
   `"no HF_TOKEN secret (fine - models are public)"` / clone unauthenticated
   when absent — the repo is public and every model pinned in `config.toml` is
   ungated, so a first run should work with **no secrets attached at all**. Only
   add them (kernel's page on kaggle.com → Add-ons → Secrets, after the first
   push creates the kernel) if a run fails on a GitHub or Hugging Face rate limit.

## Day to day

**One command, from the repo root:**

```bash
kaggle/run.sh m4b_regrade          # push, poll every 30s, download output when done
kaggle/run.sh m4c_baseline_7b
```

Output lands in `logs/kaggle-pulls/<name>/` (gitignored, same as `logs/*`).

Or do it by hand with the same three steps the script wraps:

```bash
python3 -m kaggle kernels push -p kaggle/m4c_baseline_7b
python3 -m kaggle kernels status khalidhasantuhin/tool-27-m4c-baseline-7b-target
python3 -m kaggle kernels logs   khalidhasantuhin/tool-27-m4c-baseline-7b-target
python3 -m kaggle kernels output khalidhasantuhin/tool-27-m4c-baseline-7b-target -p logs/kaggle-pulls/m4c_baseline_7b
```

If you ever need to reattach to a run without pushing a new version (e.g. after
`run.sh` was interrupted, or the `id` needed fixing — see the gotcha below):

```bash
kaggle/run.sh --no-push m4c_baseline_7b
```

**Run `m4b_regrade` first** — it's the cheap one (~15-20 min, judge-only, no
new target generations) and proves the whole path (auth, dataset access, GPU
allocation) before you commit to `m4c_baseline_7b`'s 2-4 hour run.

## The one external dependency: the 3B baseline dataset

`m4b_regrade` reads the existing 3B baseline transcript from a Kaggle dataset
owned by Mehemud's account (`kmazd1110/baseline-asr-llm-jailbreak`), already
wired into that folder's `dataset_sources`. **You said it's about to go
public** — once it is, `kaggle/run.sh m4b_regrade` just works. If you push
before that and it's still private/unshared, the push itself will succeed but
the run will fail unable to find `/kaggle/input/...` — check
[https://www.kaggle.com/datasets/kmazd1110/baseline-asr-llm-jailbreak](https://www.kaggle.com/datasets/kmazd1110/baseline-asr-llm-jailbreak) under
your own login if that happens.

`m4c_baseline_7b` has no such dependency (it generates fresh) — it's runnable
right now.

## One thing to verify on your first real run

`kernel-metadata.json` requests `"machine_shape": "NvidiaTeslaT4"` for GPU
T4 x2 (Kaggle's API has a history of silently defaulting to a single P100
instead — see the CLI's GitHub issue #821). `kaggle/run.sh`'s status polling
can't see which accelerator you actually got; check the run's page on
kaggle.com once, and if it landed on P100, that's a Kaggle-side API quirk, not
something fixable from this repo.

## Gotcha we actually hit: `id`'s slug must match the title's slug

The `id` field in `kernel-metadata.json` isn't authoritative by itself — Kaggle
derives the real kernel slug from the **title** (lowercased, spaces → dashes,
punctuation stripped) and silently uses *that* instead if it disagrees with
`id`, only warning `"Your kernel title does not resolve to the specified id"`.
This happened on the very first push here: title `"Tool 27 - M4c baseline (7B
target)"` produced the slug `tool-27-m4c-baseline-7b-target`, not the
`tool27-m4c-baseline-7b` the `id` field asked for — so the kernel came up at
`khalidhasantuhin/tool-27-m4c-baseline-7b-target`, and `id` was fixed to match
after the fact (that's why it looks like a mouthful — it's the title's exact
slug, not a choice).

`m4b_regrade`'s title was changed up front to the plain string
`tool27-m4b-regrade` (no spaces, no punctuation) specifically so its slug is
guaranteed to equal its `id` with no surprises.

**If this bites you again:** the push still succeeds and the kernel still
runs — only the address is different from what you expected. Read the printed
URL, fix `id` in that folder's `kernel-metadata.json` to match it, then use
`kaggle/run.sh --no-push <name>` to resume monitoring instead of pushing again
(pushing again just creates a redundant version of the same kernel).

## Why not just edit-and-click-run on kaggle.com?

You still can — nothing here stops that. This just lets you trigger and collect
runs without leaving the VS Code terminal. The actual logic (`attacks/`, `defense/`,
`core/`, `report.py`, `regrade.py`) already lives in the repo and is what these thin
notebooks `git clone` and drive; pushing via the CLI doesn't change that, it just
automates the "open kaggle.com, paste, click Run, wait, download" loop.
