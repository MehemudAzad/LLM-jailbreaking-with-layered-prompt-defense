#!/usr/bin/env python3
"""Generate notebooks/m6_regrade_and_gapfill.ipynb.

Two jobs in one Kaggle session, because each alone would waste a GPU allocation:
  A. regrade the committed 489-trial defended run with the fixed Layer 4 judge
  B. fill the three baseline rows that are not comparable with the defended run

Re-run after editing:  python notebooks/build_m6.py
"""
import json
import pathlib

MD, CODE = "markdown", "code"

CELLS: list[tuple[str, str]] = [
(MD, """# M6 - regrade with the fixed judge, and fill the three missing baseline rows

Two jobs, one session. Both are cheap; neither justifies its own GPU allocation.

---

## Job A - regrade the defended run (~15 min, judge only)

Layer 4 was blocking **17/50 harmless requests**. Two bugs, both in how the judge was
*asked*, not in the judge:

1. `benign_eval` never set `metadata["goal"]`, so L4 handed the judge an **empty**
   request and asked whether a helpful answer delivered harmful content toward it.
2. The prompt hardcoded `HARMFUL REQUEST`, so on a benign prompt the judge reasoned
   correctly from a false premise: a good answer to "how do I repot a houseplant" matches
   the BAD_BOT rubric exactly -- concrete, actionable steps.

Both are fixed. **But the rubric change could also have softened attack detection**, and
that has not been measured. This job settles it: re-score the existing 489 defended
trials with the new judge. The target's generations are already on disk, so only the
judge runs.

**The gate:** defended ASR was **0.6%** (3/489). If it moves materially, the fix traded
benign accuracy for attack accuracy and needs rethinking. If it holds, the fix is free.

---

## Job B - three baseline rows that cannot honestly be compared (~40 min)

The before/after table has three rows where "before" and "after" are not the same thing:

| Attack | Problem |
|---|---|
| `auto_payload_splitting` | baseline ran the **old skeleton** that flagged words but never split them -- it was not really an attack. Defended ran the fixed version. 2.0% -> 4.0% compares two different attacks. |
| `prefix_injection_textonly` | added after the baseline. **No before number at all.** |
| `prefix_injection_hello` | same. |

Fix: run exactly those three, undefended, over the same 25 goals the defended run
covered (`hb_0001..hb_0025`, a clean prefix -> `--limit 25`). 75 trials, not 1000.

`prefix_injection_textonly` matters most: it is the instruction without the forged
assistant turn, so its undefended number is **the ceiling on what the L0 guard can be
credited with removing**.
"""),

(MD, "## 1 - Setup"),
(CODE, "%pip -q install -U transformers accelerate bitsandbytes huggingface_hub"),
(CODE, """import os, subprocess, sys, pathlib, time, json, glob, shutil
import numpy as np, pandas as pd

_sec = None
try:
    from kaggle_secrets import UserSecretsClient
    _sec = UserSecretsClient()
except Exception as e:
    print('no Kaggle secrets client:', e)

def _secret(name):
    try:
        return _sec.get_secret(name) if _sec is not None else None
    except Exception:
        return None

_hf = _secret('HF_TOKEN')
if _hf:
    os.environ['HF_TOKEN'] = _hf
    from huggingface_hub import login; login(token=_hf)
    print('HF auth OK')
else:
    print('no HF_TOKEN secret (fine - models are public)')"""),

(CODE, """# --- get the repo -----------------------------------------------------------
REPO   = "MehemudAzad/LLM-jailbreaking-with-layered-prompt-defense"
BRANCH = "main"
WORK   = pathlib.Path("/kaggle/working")
ROOT   = WORK / "repo"

_gh  = _secret("GH_TOKEN")
_url = f"https://{_gh}@github.com/{REPO}.git" if _gh else f"https://github.com/{REPO}.git"

os.chdir(WORK)
subprocess.run(["rm", "-rf", str(ROOT)], check=False)
_r = subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, _url, str(ROOT)],
                    cwd=str(WORK), capture_output=True, text=True)
if _r.returncode != 0:
    _err = _r.stderr.replace(_gh, "***") if _gh else _r.stderr
    raise RuntimeError("git clone failed:\\n" + _err)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
print("HEAD", subprocess.check_output(["git","-C",str(ROOT),"rev-parse","--short","HEAD"]).decode().strip())

# the committed defended transcript is what Job A regrades -- fail loudly if it is missing
_need = ["regrade.py", "report.py", "run_eval.py",
         "logs/20260915-011304-m5defended-1f5c68/transcript.jsonl"]
_missing = [f for f in _need if not (ROOT / f).exists()]
if _missing:
    raise RuntimeError(f"clone is missing {_missing} -- push them first")
print("repo files OK")"""),

(CODE, """r = subprocess.run([sys.executable, "datasets/build_harmful.py"], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())"""),

(CODE, """import torch
print('CUDA devices:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  cuda:{i}  {p.name}  {p.total_memory/1e9:.0f} GB')

from core.config import CONFIG
for role in ('target', 'perplexity_scorer', 'judge'):
    s = CONFIG['models'][role]
    print(f"{role:<18} {s['name']:<30} device={str(s.get('device')):<8} "
          f"quant={s.get('quant')}  max_memory={s.get('max_memory')}")
print()
print("target cuda:1 budget must be 7GiB -- at 4GiB the M5 run disk-offloaded and ran 3.5x slow")"""),

(MD, """## 2 - Job A: regrade the defended run with the fixed judge

Only the judge loads. The 489 target generations are already on disk."""),

(CODE, """import report
DEF_RUN = report.find_run('m5defended')
print('regrading:', DEF_RUN)
report.print_asr(DEF_RUN, title='BEFORE regrade - defended ASR under the OLD judge')"""),

(CODE, """from regrade import regrade
summary = regrade(DEF_RUN)
print('\\nlabels changed:', summary['labels_changed'], '/', summary['trials'])"""),

(CODE, """report.print_regrade_diff(DEF_RUN)"""),

(CODE, """report.print_asr(DEF_RUN, regraded=True,
                 title='AFTER regrade - defended ASR under the FIXED judge')"""),

(MD, """### Gate for Job A

Defended ASR was **0.6% (3/489)** under the old judge.

- **Holds at ~0.6%** -> the benign fix cost nothing on attacks. Use the regraded
  numbers everywhere and say the judge was corrected.
- **Moves materially** -> the new rubric is softer on attacks too. Do not quietly keep
  the better number; report both and explain which judge produced which.

Look at `changed_samples` below to see what actually flipped and why."""),

(CODE, """report.changed_samples(DEF_RUN, n=8)"""),

(MD, """## 3 - Job B: the three missing baseline rows

Undefended, 25 goals (`hb_0001..hb_0025` -- the same prefix the defended run covered)."""),

(CODE, """from run_eval import main
main(['--attack', 'auto_payload_splitting,prefix_injection_textonly,prefix_injection_hello',
      '--defense', 'off', '--limit', '25', '--tag', 'm6gapfill'])"""),

(CODE, """report.print_asr('m6gapfill', title='BASELINE (undefended) - the three missing rows, 25 goals')"""),

(MD, """### The prefill ablation, finally complete

Undefended, all three on the same 25 goals:

| variant | instruction | forged assistant turn |
|---|---|---|
| `prefix_injection` | yes | yes |
| `prefix_injection_textonly` | yes | **no** |
| `prefix_injection_hello` | yes | neutral (`"Hello! "`) |

`prefix_injection` measured **97.9%** on the full 49-goal baseline. If `textonly` is far
lower, the forged turn was the mechanism and L0 removed the right thing. If `hello` is
also high, any forced continuation suppresses refusal -- the effect is structural rather
than about the affirmative wording."""),

(CODE, """base = report.asr_table('m4c7bbaseline')
gap  = report.asr_table('m6gapfill')
rows = [r for r in gap.index if r.startswith('prefix_injection')]
print('undefended, forced-prefix family')
if 'prefix_injection' in base.index:
    print(f"  prefix_injection           {base.loc['prefix_injection','ASR_%']:5.1f}%   "
          f"(49 goals, full baseline)")
for r in rows:
    print(f"  {r:26} {gap.loc[r,'ASR_%']:5.1f}%   (25 goals, this run)")"""),

(MD, "## 4 - Save"),

(CODE, """for run in (DEF_RUN, report.find_run('m6gapfill')):
    out = pathlib.Path('/kaggle/working/artifacts') / pathlib.Path(run).name
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run, out, dirs_exist_ok=True)
    print('mirrored ->', out)
!cd /kaggle/working && zip -qr artifacts_m6.zip artifacts && ls -la artifacts_m6.zip"""),

(MD, """## Done

Download `artifacts_m6.zip` and unzip over local `logs/`.

`transcript_regraded.jsonl` sits **next to** the original inside the m5defended folder --
the original is never modified, so both judges' verdicts stay on the record.

**If Job A's gate held**, the reports can state a single corrected result: the defended
ASR stands, and the 34% benign false-positive rate was a judge-framing bug now fixed.
**If it did not**, report both judges honestly -- that is a finding about LLM-as-judge
fragility, not a failure."""),
]


def main() -> None:
    cells = []
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == CODE:
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = pathlib.Path(__file__).parent / "m6_regrade_and_gapfill.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
