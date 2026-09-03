# Tool 27 — Workplan

LLM Jailbreak Battery + Layered Prompt Defense (CSE-406, 2026). Solo project.

Status legend: ✅ done · 🔧 code done, not run · ⬜ not started

---

## 1. Context

- **Papers:** Wei et al. 2023 (battery + the two failure modes), Jain et al. 2023 (perplexity + paraphrase defenses), Alon & Kamfonas 2023 (perplexity + length classifier).
- **Grading:** Design Report 20% · Implementation + live demo 60% · Final Report 20% · **+10% bonus** for a designed & implemented defense.
- **Schedule:** per `tool27-jailbreak-battery-workflow-2.html`, both reports are due together (~week 13 Sat) and must draw from **one** evaluation pass run by mid-week-12. Build breadth-first.
- **Compute:** all GPU work on Kaggle (T4, or T4×2). One notebook per milestone; it clones the repo and drives it. Local machine (no GPU) only runs `--dry-run` and tests.
- **Repo:** `github.com/MehemudAzad/LLM-jailbreaking-with-layered-prompt-defense` (public).

## 2. Fixed decisions — `config.toml`, seed 1337

| role | model | placement | notes |
|---|---|---|---|
| target ("victim") | `Qwen/Qwen2.5-3B-Instruct` | cuda:0 | text-only, ungated, real system slot |
| perplexity scorer (L1) | `openai-community/gpt2-large` | cuda:1 | MIT, ~2 GB |
| paraphraser (L2) | **TBD (M5)** | — | must differ from target |
| judge (L4) | `Qwen/Qwen3.5-9B` | cuda:1 | nf4 4-bit, thinking off |
| helper LM (auto_* attacks) | **TBD (M6)** | — | or drop those attacks |

- **Harmful set:** AdvBench (Zou et al. 2023), seeded 50-row sample. Git-ignored; rebuilt by `datasets/build_harmful.py`.
- **Benign set:** 50 hand-written imperatives mirroring AdvBench style (FPR control + quality-drop measurement).

## 3. Attack battery — `attacks/` (17 techniques)

| category | techniques |
|---|---|
| encoding & obfuscation | `base64`, `rot13`, `leetspeak`, `disemvowel` |
| injection & formatting | `prefix_injection`, `refusal_suppression`, `style_injection_json`, `distractors` |
| persona & roleplay | `evil_confidant`, `aim`, `dev_mode`, `wikipedia_article` |
| model-assisted | `auto_payload_splitting`, `auto_obfuscation` *(need a helper LM — M6)* |
| combination | `combination_1`, `combination_2`, `combination_3` |
| control | `passthrough` |
| strategy | `adaptive` *(run whole battery per goal; success = any member is BAD_BOT — M6)* |

*Open: sprint doc suggested trimming to 8–10. Keep 17 or trim?*

## 4. Defense stack — `defense/` (order fixed in `defense/__init__.py`)

| layer | what it does | source | status |
|---|---|---|---|
| L1 perplexity | reject prompts whose worst token-window perplexity exceeds a calibrated threshold | Jain; Alon & Kamfonas | ✅ threshold 424.8, 0% benign FPR, catches obfuscation only (3/16) |
| L2 paraphrase | rewrite every prompt through a separate model before it reaches the target | Jain | backend TBD (M5) |
| L3 system hardening | instruction-hierarchy reminder + refusal priming in the system prompt | **own design** | prompt written, needs tuning (M5) |
| L4 response check | judge labels the reply BAD_BOT / GOOD_BOT / UNCLEAR; blocks BAD_BOT | **own design** + Wei rubric | 🔧 code done (M4) |
| L1.5 structural check | regex for base64 / hex / cipher runs that L1's perplexity misses | **own design (bonus)** | M7 |

---

## 5. Milestones

### M0 — Scaffold ✅

Repo, `config.toml`, `core/` (config, seed, models, datasets, transcript), `attacks/` (registry + base + 17 stubs), `defense/` (base + pipeline + 4 layer stubs), `run_eval.py`, `tests/`. `python run_eval.py --dry-run` walks the whole pipeline with a fake backend — no GPU.

### M1 — Target backend ✅

`TransformersModelHandle.generate()` — chat template, greedy decode, system-fold for templates without a system slot, dtype-auto (bf16/fp16), best-effort prefill.

- **Deliverable:** deterministic `send_prompt`; notebook `m1_model_backend.ipynb` — 3 checks pass (determinism / substantive benign answer / refusal on an obvious harmful ask).
- **Pin** the target revision in `config.toml`.

### M2 — Layer 1 perplexity, method ✅

`perplexity(text, window=N)` — teacher-forced NLL → perplexity; `window` returns the worst (highest-perplexity) contiguous N-token span.

- **Notebook** `m2_perplexity_filter.ipynb` — score benign set, per-technique catch/miss, GPT-2 vs Qwen-self-perplexity.
- **Finding:** the scorer model determines what gets caught — GPT-2 catches character obfuscation; the target-as-scorer catches instruction anomalies; neither catches both.

### M3 — Datasets + L1 recalibration ✅

- `datasets/build_harmful.py` — AdvBench → seeded 50. Benign set → 50 matched imperatives.
- **Notebook** `m3_calibrate_layer1.ipynb` — recalibrate on the **clean AdvBench-50** (Jain's reference set): threshold = max windowed-perplexity = **424.8**, 0% benign FPR, catches 3/16 (`rot13`, `leetspeak`, `disemvowel`).
- **Finding for the report:** harmful and benign *natural-language* prompts are perplexity-indistinguishable (identical medians). L1 is a gibberish detector, not a jailbreak detector — Layers 2–4 carry the fluent attacks.
- `config.toml`: L1 `enforce = true`, `threshold = 424.8`; scorer revision pinned.

### M4 — Judge (Layer 4) + baseline ASR 🔧

- `Qwen/Qwen3.5-9B` nf4 4-bit judge on cuda:1; `core/models.py` gains `quant` / `device` / `thinking` spec keys.
- `defense/layer4_response_classifier.py` — real judge prompt (Wei rubric), refusal-string pre-filter, `<think>` stripping, robust label parse.
- `run_eval.py` — `--grade` (judge labels replies even with `--defense off`); summary gains `asr_by_attack`.
- **Notebook** `m4_baseline_asr.ipynb` (needs **GPU T4 ×2**): setup → sanity pass (10 goals) → **gate: do the judge labels match the replies?** → full 50 goals, undefended.
- **Deliverable:** per-attack **baseline ASR** table — the "before" half of every result. Pin the judge revision.

### M5 — Layer 2 paraphraser + Layer 3 + defended ASR ⬜ (next)

- **L2:** real paraphraser backend. Pick the model and fit it in VRAM alongside target + judge + scorer (likely 4-bit, or a smaller paraphraser). Fixed instruction `"paraphrase the following sentences:"`; replace `ctx.prompt` with the paraphrase, keep the original in metadata.
- **L3:** tune the hardening system prompt; measure its benign-quality cost.
- **Notebook** `m5_defended_asr.ipynb`:
  1. **benign quality-drop** — 50 benign prompts, defended vs undefended (refusal rate, answer length, a quality proxy).
  2. **defended ASR pass** — full battery, `--defense on`, 50 goals → per-attack defended ASR.
  3. **before/after table** + per-layer attribution (which layer blocked each attack, from the recorded `verdicts`).
- **Deliverable:** defended ASR table + cost-of-defense numbers + per-layer credit. **The core result of the project.**

### M6 — Adaptive attack + helper LM ⬜

- Helper LM backend for `auto_payload_splitting` / `auto_obfuscation` — or drop those two (decide).
- `attacks/adaptive.py` battery loop + `run_eval` support for `--attack adaptive`: per goal, run every technique; success if **any** member is BAD_BOT.
- **Notebook** `m6_adaptive.ipynb` — adaptive ASR, undefended vs defended.
- **Deliverable:** the headline single number — "the battery achieves X% against the undefended target, Y% against the full stack."

### M7 — Own design / bonus ⬜

- **L1.5 structural check** — regex for long base64 / hex / cipher-looking runs; closes the base64 gap M3 found, at ~0 cost.
- L3 / L4 tuning from M5 & M6 results; re-run affected evaluations.
- **Deliverable:** the finalized defense stack + the written design justification for the +10% bonus.
- **Then freeze everything** and run the **one definitive evaluation pass** that both reports cite.

### M8 — Reports + demo ⬜

- **Design Report (20%):** attack definitions + per-technique prompt-structure breakdown; system / data-flow diagram; pipeline diagram (baseline vs each layer); justification — why the layers cover **both** failure modes (competing objectives + mismatched generalization) with the papers' cited numbers.
- **Final Report (20%):** steps + snapshots; is it successful, why / why not; observed output at attacker / target / gateway; countermeasure writeup (bonus).
- **Demo (part of the 60%):** script it; rehearse twice on the pinned setup.
- Both reports draw from the single M7 evaluation pass.

**Report-template translation** (network-attack template → Tool 27):

| template asks for | Tool 27 equivalent |
|---|---|
| topology diagram | data-flow: battery script → [L1 → L2 → L3] → target → L4 → response |
| timing diagram | pipeline sequence diagram: undefended request vs the same prompt through each layer |
| packet / frame details | per-attack prompt structure (prefix / suffix / encoding / persona) + how each layer transforms it |
| justification | coverage of both failure modes, with cited numbers from the three papers |
| attacker PC / victim / server | battery script / pinned target model / paraphraser + detector gateway |
| snapshots of victim screen | captured request/response transcripts, pre- and post-defense |
| countermeasure | L3 + L4 + L1.5, written up as the bonus-eligible defense |

---

## 6. Open decisions

1. **Battery size** — keep all 17 techniques, or trim to ~10 (per the sprint doc)?
2. **M5 paraphraser** — which model, and how it fits in VRAM alongside target + judge + scorer?
3. **M6 `auto_*` attacks** — add a helper LM, or drop them?
4. **Calendar** — actual dates? (the workflow page assumes "now = week 10, due week 13 Sat")
5. **Cross-check target** (optional, week 12) — re-run the final pass against Gemma-3-4b or Mistral-7B to show the result isn't model-specific?

---

## 7. Progress

| milestone | status |
|---|---|
| M0 scaffold | ✅ |
| M1 target backend | ✅ |
| M2 Layer 1 method | ✅ |
| M3 datasets + recalibration | ✅ |
| M4 judge + baseline ASR | 🔧 run pending |
| M5 L2 + L3 + defended ASR | ⬜ |
| M6 adaptive + helper LM | ⬜ |
| M7 own design / bonus + freeze | ⬜ |
| M8 reports + demo | ⬜ |
