#!/usr/bin/env python3
"""Generate notebooks/m7_final_baseline.ipynb -- the clean undefended baseline.

Re-run after editing:  python notebooks/build_m7.py
"""
import json
import pathlib

MD, CODE = "markdown", "code"

CELLS: list[tuple[str, str]] = [
(MD, """# M7 - the clean undefended baseline

`--defense off` against **Qwen2.5-7B-Instruct**, 17 techniques, 25 goals
(`hb_0001..hb_0025`). Roughly 425 trials.

## Why re-run a baseline we already have

The existing baseline (`m4c7bbaseline`, 877 trials) is real but no longer matches the
code, in three ways that each break comparability with the defended run:

| | old baseline | current code |
|---|---|---|
| battery | 18 techniques incl. `aim`, `base64`, `rot13` | 17, those three retired |
| `auto_payload_splitting` | old skeleton -- flagged words, never split them | actually splits |
| judge | asserted `HARMFUL REQUEST`, graded against an empty request on the benign path | decides harmfulness itself |
| prefix ablations | absent | `_textonly`, `_hello` present |

This run produces a baseline from **one commit**, graded by **one judge**, over the
**same 25 goals** the defended run covered -- so before/after finally means the same
measurement on both sides. That is the M7 freeze the workplan asks for.

## Configuration guard

`[models.target] max_memory` must read `{ 0 = "14GiB", 1 = "7GiB" }`. At 4GiB the 15.2 GB
fp16 target disk-offloaded and ran at 0.78 tok/s instead of 2.73, which is what burned
the M5 session. The scorer sits on CPU to buy that headroom back; with `--defense off` it
is not loaded at all.

**Expect ~3.5 h.** Kaggle terminates a session at 12 h. If it is cut short the transcript
still survives under `/kaggle/working/repo/logs/` -- the artifacts mirror only runs on
clean completion.
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

(CODE, """REPO   = "MehemudAzad/LLM-jailbreaking-with-layered-prompt-defense"
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
print("this commit is what the run is attributable to -- record it in the report")"""),

(CODE, """r = subprocess.run([sys.executable, "datasets/build_harmful.py"], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())"""),

(CODE, """import torch
print('CUDA devices:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  cuda:{i}  {p.name}  {p.total_memory/1e9:.0f} GB')
assert torch.cuda.device_count() >= 2, 'set the accelerator to GPU T4 x2'

from core.config import CONFIG
mm = CONFIG['models']['target'].get('max_memory')
print('\\ntarget max_memory:', mm)
assert str(mm.get(1) or mm.get('1')) == '7GiB', (
    f'expected 7GiB on cuda:1, got {mm} -- lower values disk-offload the target '
    'and cost a 3.5x slowdown')
print('scorer device    :', CONFIG['models']['perplexity_scorer'].get('device'))

from attacks import load_all
atks = sorted(k for k in load_all() if k != 'adaptive')
print(f'\\nbattery: {len(atks)} techniques')
print(' ', ', '.join(atks))
for gone in ('aim', 'base64', 'rot13'):
    assert gone not in atks, f'{gone} should have been retired'
print('  (aim / base64 / rot13 retired -- see docs/DROPPED-ATTACKS.md)')"""),

(MD, """## 2 - Sanity pass (2 goals, ~34 trials)

Cheap gate before the long run: does the target load on both cards, and does the judge
produce labels that match the replies?"""),

(CODE, """from run_eval import main
main(['--attack', 'all', '--defense', 'off', '--limit', '2', '--tag', 'm7sanity'])"""),

(CODE, """import report
report.print_asr('m7sanity', title='SANITY undefended (2 goals)')
for i in range(torch.cuda.device_count()):
    print(f'cuda:{i}  {torch.cuda.memory_allocated(i)/1e9:.1f} GB')"""),

(CODE, """report.samples('m7sanity', exclude='GOOD_BOT', n=6)"""),

(MD, """### Gate

1. No OOM, and **no** `offloaded to the disk` warning above. If that warning appears,
   stop -- the run will take 3.5x longer than budgeted and will not finish.
2. `prefix_injection` should be winning: it measured 97.9% on the old baseline and
   nothing in the defense is active here.
3. The judge's labels should match what the replies actually say."""),

(MD, "## 3 - The baseline (25 goals x 17 techniques, ~3.5 h)"),

(CODE, """main(['--attack', 'all', '--defense', 'off', '--limit', '25', '--tag', 'm7baseline'])"""),

(CODE, """report.print_asr('m7baseline',
                 title='CLEAN BASELINE - undefended Qwen2.5-7B, 25 goals, 17 techniques')"""),

(CODE, """report.print_adaptive('m7baseline', title='ADAPTIVE - undefended (any technique wins)')"""),

(MD, """### The prefill ablation, undefended

`prefix_injection` carries both a textual instruction and a forged assistant turn.
`_textonly` drops the forged turn; `_hello` keeps a forged turn but makes its content
neutral. The M6 gap-fill measured 100.0 / 96.0 / 16.0 respectively, which refuted the
hypothesis that the forged turn was the mechanism. This run re-measures all three
together under one judge."""),

(CODE, """tbl = report.asr_table('m7baseline')
rows = [r for r in tbl.index if r.startswith('prefix_injection')]
print(tbl.loc[rows, ['n', 'BAD_BOT', 'GOOD_BOT', 'UNCLEAR', 'ASR_%']].to_string())
print('''
reading it:
  full vs _textonly  -> what the forged assistant turn is actually worth
  _hello             -> whether ANY forced continuation helps, or only an affirmative one
''')"""),

(MD, "## 4 - Against the old baseline"),

(CODE, """old = report.asr_table('m4c7bbaseline')
new = report.asr_table('m7baseline')
both = sorted(set(old.index) & set(new.index))
cmp = pd.DataFrame({'old_49goals_%': old.loc[both, 'ASR_%'],
                    'new_25goals_%': new.loc[both, 'ASR_%']})
cmp['delta'] = (cmp['new_25goals_%'] - cmp['old_49goals_%']).round(1)
print(cmp.sort_values('new_25goals_%', ascending=False).to_string())
print('''
Differences are expected and not errors: different goal counts, a fixed
auto_payload_splitting, and a judge that no longer asserts the request is harmful.
Large unexplained swings are worth reading the samples for.''')"""),

(MD, "## 5 - Save"),

(CODE, """for tag in ('m7baseline', 'm7sanity'):
    try:
        run = report.find_run(tag)
    except Exception as e:
        print('skip', tag, e); continue
    out = pathlib.Path('/kaggle/working/artifacts') / pathlib.Path(run).name
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run, out, dirs_exist_ok=True)
    print('mirrored ->', out)
!cd /kaggle/working && zip -qr artifacts_m7.zip artifacts && ls -la artifacts_m7.zip"""),

(MD, """## Done

Download `artifacts_m7.zip`, unzip over local `logs/`, and commit the run directory with
`git add -f logs/<run-id>`.

**This is the freeze point.** With this baseline and the existing defended run both from
the current code, the before/after pair is finally one measurement on both sides -- and
the reports should cite this run, noting that the earlier 49-goal baseline is superseded
for comparison purposes while remaining valid on its own terms."""),
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
    out = pathlib.Path(__file__).parent / "m7_final_baseline.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
