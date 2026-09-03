# Tool 27 — LLM Jailbreak Battery + Layered Prompt Defense

CSE-406 Project 2026. One repo, two halves, built in parallel.

Reference material is in [`docs/`](docs/): `jailbreak-paper.pdf` (Wei et al. 2023),
`baseline-defenses-for-adversarial-attacks.pdf` (Jain et al. 2023),
`detecting-language-model-attacks.pdf` (Alon & Kamfonas 2023),
`tool27-jailbreak-battery-workflow-2.html` (the working plan),
`sprint1-getting-started.md` (this sprint's plan).

## Who owns what — the rule that keeps us out of each other's way

| Area | Owner | Files | Touched by the other half? |
|------|-------|-------|----------------------------|
| Attack battery | **Member A** | `attacks/*`, `datasets/*` | never |
| Layered defense | **Member B** | `defense/*` | never |
| Shared core + harness | **joint, frozen after Week 10** | `core/*`, `run_eval.py`, `config.toml`, `attacks/base.py`, `defense/base.py` | only by agreement |
| Raw transcripts | both, append-only | `logs/*` | no conflicts — one dir per run |

The two `base.py` files and `core/` are the **contract**. Agree them once, then treat
them as frozen. Everything else is additive:

- Member A drops **one file per technique** into `attacks/` and decorates the class with
  `@register`. Nothing outside `attacks/` + `datasets/` changes.
- Member B fills in the **four files** in `defense/` (they start as pass-through stubs).
  Nothing outside `defense/` changes.
- `logs/` only ever gains new directories. Never edit an existing transcript — re-run.

## Layout

```
config.toml                     pinned models + seed + thresholds + dataset paths
run_eval.py                     the harness: attacks x defense -> logs/<run-id>/transcript.jsonl
core/
  config.py                     loads config.toml (+ config.local.toml overrides)
  seed.py                       seed_everything()
  models.py                     ModelHandle contract; FakeModelHandle (real backend = TODO)
  datasets.py                   load_harmful() / load_benign()
  transcript.py                 append-only JSONL logger
attacks/
  base.py                       AttackTechnique ABC + AttackOutput          [FROZEN]
  __init__.py                   @register decorator + load_all() registry   [FROZEN]
  passthrough.py                control condition (baseline ASR)
  base64_encode.py rot13.py leetspeak.py disemvowel.py           encoding & obfuscation
  prefix_injection.py refusal_suppression.py style_injection_json.py distractors.py   injection & formatting
  evil_confidant.py aim.py dev_mode.py wikipedia_article.py      persona & roleplay
  auto_payload_splitting.py auto_obfuscation.py                  model-assisted
  combination.py adaptive.py                                     combination & adaptive
defense/
  base.py                       DefenseLayer ABC + DefenseContext + Verdict [FROZEN]
  pipeline.py __init__.py       LayeredDefense: pre-layers -> target -> post-layers
  layer1_perplexity_filter.py   Jain et al. / Alon & Kamfonas   (stub: scores + logs, does not block yet)
  layer2_paraphrase.py          Jain et al.                     (stub: pass-through)
  layer3_system_hardening.py    our design (bonus-eligible)
  layer4_response_classifier.py our design + automated grader
datasets/
  harmful_behaviors.jsonl       FROZEN eval set (schema stub -> populate from AdvBench)
  benign_prompts.jsonl          FROZEN benign set for calibration + quality-drop
logs/                           raw transcripts, one dir per run
tests/test_wiring.py            smoke test: everything registers + a dry run returns 0
docs/                           the three papers, the workflow page, the sprint plans
```

## Quick start

```bash
python3 --version                     # need >= 3.11
python run_eval.py --dry-run --limit 1 # fake models, walks attacks x defense, writes a transcript
pytest -q                             # wiring smoke test  (pip install pytest)
```

`--dry-run` forces `backend = "fake"` for every model, so the whole pipeline runs
end-to-end with no weights, no GPU, no downloads. This is the only thing meant to run on
the laptop.

## Running for real (Kaggle)

Real model runs happen in Kaggle notebooks (T4 GPU) — see `notebooks/`. Each milestone
is one notebook that `git clone`s this repo and drives it:

| Notebook | Milestone | Status |
| --- | --- | --- |
| `m1_model_backend.ipynb` | target model (`Qwen/Qwen2.5-3B-Instruct`) via `TransformersModelHandle` | ready to run |
| `m2_perplexity_filter.ipynb` | defense Layer 1 — `gpt2-large` scorer, calibrated windowed-perplexity threshold | ready to run |
| _(m3)_ | judge model + Layer 4, then baseline ASR once AdvBench is frozen | next |

Once M1 is validated and the target `revision` is pinned in `config.toml`:

```bash
python run_eval.py --attack all --defense off   # baseline ASR against the undefended target
python run_eval.py --attack all --defense on    # the full evaluation pass
```

## Pipeline

```
goal ─▶ attack.apply() ─▶ [ L1 perplexity ─▶ L2 paraphrase ─▶ L3 hardening ] ─▶ target LLM ─▶ [ L4 response check ] ─▶ verdict
                             pre-target layers (short-circuit on block)                          post-target layer
```

## Reproducibility checklist (pin before the first real run)

- [ ] `seed` set in `config.toml`
- [ ] every `[models.*]` has a real `name` **and** a pinned `revision` (no `PIN-ME`)
- [ ] `datasets/harmful_behaviors.jsonl` frozen; `harmful_count` filled in
- [ ] `datasets/benign_prompts.jsonl` frozen; `benign_count` filled in
- [ ] model backend deps uncommented **and version-pinned** in `requirements.txt`
- [ ] one full evaluation pass saved under `logs/` and committed as evidence
