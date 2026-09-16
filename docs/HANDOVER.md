# Handover — what is done, what is left

Read this first. It is the honest state of the project, including the parts that are
wrong and the traps that will waste your GPU quota if you do not know about them.

---

## 🚨 Read before you run anything

### 1. `config.toml` has a 3.5× slowdown booby-trap

```toml
[models.target]
max_memory = { 0 = "14GiB", 1 = "4GiB" }   # ← THIS
```

Cutting cuda:1 from 7 GiB to 4 GiB was done to squeeze a 4th model (the paraphraser)
onto the two T4s. It forced the 7B target into **partial CPU offload**:

| Run | cuda:1 budget | Models resident | Throughput |
|---|---|---|---|
| M4c baseline | 7 GiB | 2 | **2.73 tok/s** |
| M5 defended | 4 GiB | 4 | **0.78 tok/s** |

That is why the M5 run needed ~15 h and got killed by Kaggle's 12 h cap at trial
489/1000. **Fix this before any re-run.** Options, cheapest first:

- Put `perplexity_scorer` on CPU (gpt2-large, only used for a short scoring pass) and
  restore target cuda:1 to 7 GiB.
- Or drop the paraphraser to a ~250 M model (e.g. a T5 paraphraser) instead of
  Qwen2.5-1.5B.
- Or run L1/L1.5 as a **separate CPU-only pre-pass** and only load target+judge on GPU.

### 2. Kaggle kills the session at 12 hours, and the mirror never fires

`TranscriptLogger.close()` writes the mirror to `/kaggle/working/artifacts`. If the
session is killed mid-run, **that never happens** — the artifacts folder will look almost
empty and you will think the run produced nothing.

**It didn't.** The real transcript is at `/kaggle/working/repo/logs/<run-id>/` inside the
session's repo clone, and it comes down with `kaggle kernels output`. That is how the
489 defended trials were recovered. Always check `repo/logs/` before concluding a run
was lost.

### 3. Three models are still unpinned

`config.toml` has `revision = "PIN-ME"` for **target**, **helper** and **paraphraser**.
The course spec explicitly requires pinning models/datasets/seeds for reproducibility.
The notebooks print the current hashes — paste them in.

---

## ✅ What is finished

| Milestone | State | Evidence |
|---|---|---|
| M0–M3 scaffold, target backend, L1 perplexity, datasets | done | `logs/`, notebooks m1–m3 |
| M4 judge + baseline ASR | done | `logs/20260912-070823-m4c7bbaseline-9e6106` (877 trials) |
| M5 defenses + defended ASR | **partial** (489/1000) | `logs/20260915-011304-m5defended-1f5c68` |
| M6 adaptive attack | done, computed from transcripts | `report.adaptive_asr()` |
| M7 bonus layers | built + validated | `defense/layer0_*`, `defense/layer1_5_*` |
| M8 reports + demo | demo done, reports stale | `demo.py`, `docs/PRESENTATION.md` |

### Headline numbers

| | Before | After |
|---|---|---|
| Overall ASR | 17.3% (152/877) | **0.6%** (3/489) |
| `prefix_injection` | 97.9% | **0.0%** |
| `distractors` | 49.0% | **0.0%** |
| Adaptive (any technique wins) | 100% | **12%** |
| Benign requests still served | 100% | **62%** |

### The defense stack (6 layers, 3 are our own design)

```
L0  prefill guard      OURS  strips a client-supplied assistant turn
L1  perplexity         Jain / Alon & Kamfonas
L1.5 structural check  OURS  decodes base64/hex payloads
L2  paraphrase         Jain
L3  system hardening   OURS
       → TARGET →
L4  response judge     OURS  BAD_BOT / GOOD_BOT / UNCLEAR
```

---

## ❌ What is broken or incomplete

### A. Layer 4 blocks 34% of harmless requests — **highest priority**

Measured: 17/50 benign prompts blocked, all by L4.

**Cause:** L4 asks the judge *"did the model comply with this harmful request?"* — but it
is handed the request unconditionally, with no check that the request was harmful in the
first place. A benign question produces a helpful answer, the judge correctly observes
"yes, it complied", and L4 blocks it.

**Fix direction:** L4 needs a harmfulness gate on the *request* before judging the
*response*, or a judge prompt that can return "request was benign". Then re-measure with
`benign_eval.py`.

This does **not** contaminate the attack numbers — there the requests genuinely were
harmful — but it makes the stack undeployable as-is, and it is the obvious next
contribution.

### B. The before/after comparison has two apples-to-oranges rows

| Attack | Problem |
|---|---|
| `auto_payload_splitting` | Baseline ran the **old skeleton** (flagged words but never split them — it wasn't really an attack). Defended ran the **fixed** implementation. 2.0% → 4.0% compares two different attacks. |
| `prefix_injection_textonly`, `prefix_injection_hello` | Added after the baseline, so they have **no before number** at all. |

Verify yourself:
```bash
python3 -c "
import json,pathlib
for d in ['20260912-070823-m4c7bbaseline-9e6106','20260915-011304-m5defended-1f5c68']:
    r=[json.loads(l) for l in open(pathlib.Path('logs')/d/'transcript.jsonl')]
    t=[x for x in r if x.get('type')=='trial' and x['attack']=='auto_payload_splitting']
    print(d, sorted(t[0]['metadata'])[:4])"
```

**Either** exclude those three rows from the before/after table and say why, **or** fix
it properly with (C).

### C. M7 freeze — the one clean run both reports should cite

The workplan calls for a **single definitive evaluation pass** that both reports cite.
Right now the baseline and defended runs used *different attack sets* (18 vs 20) and
*different code versions*. That is defensible if stated, but weaker than a clean pair.

The proper fix: after (A) and the memory fix, re-run **both** arms from the same commit:

```bash
python run_eval.py --attack all --defense off --tag final-baseline
python run_eval.py --attack all --defense on  --tag final-defended
```

Budget: at 2.73 tok/s that is roughly 8 h per arm for 50 goals × 20 attacks. **Split
across two Kaggle sessions**, or cut to 25 goals per arm (which is what the current
defended run effectively is anyway).

### D. Reports

- `docs/design-report.tex` — still describes the **3B** target and says M4 has not run.
  Needs the 7B name, the real numbers, and the L0/L1.5 layers added to the diagrams.
- **Final report** — not started. Needs: steps + snapshots, was the attack successful and
  why, observed output at attacker/victim/gateway, countermeasure writeup (the bonus).
- **Contribution split** — the spec says *"Clearly specify in the report which member was
  responsible for each part."* The design report lists both names but no split. Required.

---

## Suggested order of work

1. **Fix the `max_memory` trap** (5 min, unblocks everything else)
2. **Fix the L4 benign bug** (A) — biggest real improvement, and a strong report finding
3. **Pin the three `PIN-ME` revisions** (spec requirement, 5 min once a notebook runs)
4. **Re-run both arms from one commit** (C) — kills problem (B) for free
5. **Update `design-report.tex`**, write the final report, add the contribution split

Steps 1–3 are cheap and make everything after them trustworthy. Do not skip to 4.

---

## How to run things

```bash
# local, no GPU — sanity check the wiring before spending quota
python3 -m pytest -q
python3 run_eval.py --dry-run --limit 1 --defense on

# the live demo (laptop only, real data, nothing can fail)
python3 demo.py

# push a notebook to Kaggle and pull results back
kaggle/run.sh m5_defended_asr
kaggle/watch.sh m5_defended_asr        # status only, safe to Ctrl-C

# read any finished run
python3 -c "import report; report.print_asr('m5defended')"
python3 -c "import report; report.print_attribution('m5defended')"
python3 -c "import report; report.print_adaptive('m5defended')"
python3 -c "import benign_eval as b; b.print_compare('logs/<und-run>','logs/<def-run>')"
```

## Where things live

| Path | What |
|---|---|
| `attacks/` | 20 techniques, one file each |
| `defense/` | the 6 layers, in pipeline order |
| `report.py` | ASR tables, per-layer attribution, adaptive ASR |
| `benign_eval.py` | cost-of-defense on the benign set |
| `demo.py` | the presentation demo, laptop-only |
| `logs/` | committed evidence — do not edit, re-run instead |
| `docs/SIMPLE-EXPLANATION.md` | plain-English tour of the whole project |
| `docs/ATTACKS-EXPLAINED.md` | every attack with a real rendered prompt |
| `docs/PRESENTATION.md` | what to say to the teacher |
