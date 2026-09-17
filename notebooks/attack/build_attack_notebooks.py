#!/usr/bin/env python3
"""Generate notebooks in notebooks/attack/:
  1) 01_download_dependencies.ipynb  (Internet ON: downloads wheels and packages repo)
  2) 02_offline_attack_eval.ipynb    (Internet OFF: installs wheels, loads local model, runs attacks)

Re-run to regenerate:
  python notebooks/attack/build_attack_notebooks.py
"""
import json
import pathlib

MD, CODE = "markdown", "code"

# ---------------------------------------------------------------------------
# Notebook 1: Download Dependencies & Package Offline Bundle
# ---------------------------------------------------------------------------
NB1_CELLS: list[tuple[str, str]] = [
    (MD, """# 01 - Download Dependencies & Package Offline Bundle
**Environment**: Kaggle GPU or CPU with **Internet ON**.

This notebook downloads all python wheels (`.whl`) and bundles the repository code + harmful prompt dataset so you can run attacks in a completely **offline (air-gapped)** Kaggle session.

---

### Workflow
1. Run this notebook with **Internet ENABLED**.
2. When finished, find `offline_attack_bundle.zip` in `/kaggle/working`.
3. In Kaggle's right sidebar under **Output**, click the three dots $\\rightarrow$ **New Dataset** (name it `offline-attack-bundle`).
4. Attach that dataset to the offline attack notebook (`02_offline_attack_eval.ipynb`).
"""),

    (MD, "## 1 - Prepare Directory Structure"),
    (CODE, """import os, sys, shutil, pathlib, subprocess

WORK = pathlib.Path("/kaggle/working") if pathlib.Path("/kaggle/working").exists() else pathlib.Path("./workspace")
BUNDLE_DIR = WORK / "offline_attack_bundle"
WHEELS_DIR = BUNDLE_DIR / "wheels"
REPO_DIR = BUNDLE_DIR / "repo"

if BUNDLE_DIR.exists():
    shutil.rmtree(BUNDLE_DIR)

WHEELS_DIR.mkdir(parents=True, exist_ok=True)
REPO_DIR.mkdir(parents=True, exist_ok=True)
print(f"Bundle directory ready at: {BUNDLE_DIR}")
"""),

    (MD, "## 2 - Download Python Wheels for Offline Installation"),
    (CODE, """# Download wheels compatible with Python 3.10 / 3.11 Linux x86_64
packages = [
    "transformers>=4.45.0",
    "accelerate>=0.26.0",
    "sentencepiece",
    "protobuf",
    "tomli",
]

cmd = [
    sys.executable, "-m", "pip", "download",
    "-d", str(WHEELS_DIR),
    *packages
]
print("Downloading wheels... this may take 1-2 minutes.")
res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode != 0:
    print("Download stderr:", res.stderr)
    raise RuntimeError("pip download failed")

downloaded_wheels = list(WHEELS_DIR.glob("*.whl"))
print(f"Successfully downloaded {len(downloaded_wheels)} wheels to {WHEELS_DIR}")
"""),

    (MD, "## 3 - Bundle Repository Code & Dataset"),
    (CODE, """# Determine repo location (either current directory or clone if needed)
current_path = pathlib.Path.cwd()
if (current_path / "run_eval.py").exists():
    src_repo = current_path
elif (current_path / "repo" / "run_eval.py").exists():
    src_repo = current_path / "repo"
else:
    # Clone from GitHub
    print("Cloning repository from GitHub...")
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/MehemudAzad/LLM-jailbreaking-with-layered-prompt-defense.git",
                    str(src_repo := WORK / "temp_clone")], check=True)

# Copy source tree components into bundle
for item in ["attacks", "core", "defense", "datasets", "config.toml", "run_eval.py", "report.py"]:
    src = src_repo / item
    dst = REPO_DIR / item
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.is_file():
        shutil.copy2(src, dst)

# Ensure harmful dataset is generated and included
harmful_gen = REPO_DIR / "datasets" / "build_harmful.py"
if harmful_gen.exists():
    subprocess.run([sys.executable, str(harmful_gen)], cwd=str(REPO_DIR), check=True)
    print("Harmful behaviors dataset generated at repo/datasets/harmful_behaviors.jsonl")

print("Repository code successfully bundled.")
"""),

    (MD, "## 4 - Create Offline Zip Archive"),
    (CODE, """archive_path = WORK / "offline_attack_bundle.zip"
if archive_path.exists():
    archive_path.unlink()

print("Creating zip archive...")
shutil.make_archive(str(WORK / "offline_attack_bundle"), "zip", root_dir=str(BUNDLE_DIR))
size_mb = archive_path.stat().st_size / (1024 * 1024)
print(f"Created {archive_path.name} ({size_mb:.1f} MB)")
"""),

    (MD, "## 5 - Sanity Check: Test Model Loading (Gemma)"),
    (CODE, """# Validate that the downloaded wheels and the local Gemma model work end-to-end
import os, sys, glob, json, pathlib, subprocess

# 1. Test installing the downloaded wheels locally
print("Testing wheel installation...")
res = subprocess.run([
    sys.executable, "-m", "pip", "install",
    "--no-index", f"--find-links={WHEELS_DIR}",
    "transformers", "accelerate", "sentencepiece", "protobuf"
], capture_output=True, text=True)
print("Wheel install output:", res.stdout[-200:] if res.stdout else res.stderr[-200:])

# 2. Check model path
TEST_PATH = pathlib.Path("/kaggle/input/models/google/gemma/pytorch/2b-it/2")

# If the path differs slightly, attempt to auto-discover
if not TEST_PATH.exists():
    for candidate in pathlib.Path("/kaggle/input").rglob("*2b-it*"):
        if candidate.is_dir() and ((candidate / "config.json").exists() or (candidate / "model.safetensors.index.json").exists()):
            TEST_PATH = candidate
            break

print(f"Target model location: {TEST_PATH}")
if not TEST_PATH.exists():
    print("⚠️ Model path not found in /kaggle/input! If you haven't attached Gemma to this session, attach it in the right sidebar under 'Models' or 'Input'.")
else:
    # 3. Add bundled repo to sys.path and run a 1-goal test
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))

    import run_eval
    from core.config import CONFIG
    CONFIG["models"]["target"]["name"] = str(TEST_PATH)
    CONFIG["models"]["target"]["revision"] = None
    CONFIG["models"]["target"]["device"] = "auto"
    CONFIG["models"]["target"]["max_memory"] = None

    cmd_args = [
        "--attack", "passthrough,prefix_injection",
        "--defense", "off",
        "--no-grade",
        "--limit", "1",
        "--tag", "gemma-sanity",
    ]

    print("\\nRunning a 1-goal test (passthrough & prefix_injection) with defense=off, no-grade...")
    run_eval.main(cmd_args)

    # 4. Read newest transcript to verify completions
    logs = sorted(glob.glob(str(REPO_DIR / "logs/*/transcript.jsonl")), key=os.path.getmtime)
    if logs:
        print("\\n" + "=" * 70)
        print("SANITY CHECK RESULTS:")
        print("=" * 70)
        with open(logs[-1], "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("type") == "trial":
                    print(f"Attack   : {d.get('attack')}")
                    print(f"Prompt   : {(d.get('prompt') or '')[:70]}...")
                    print(f"Response : {(d.get('response') or '')[:150]}...")
                    print("-" * 70)
        print("\\n✅ Model loaded and generated responses successfully!")
        print("You can now safely create the Kaggle dataset from 'offline_attack_bundle.zip'!")
"""),
]


# ---------------------------------------------------------------------------
# Notebook 2: Offline Attack Evaluation Notebook
# ---------------------------------------------------------------------------
NB2_CELLS: list[tuple[str, str]] = [
    (MD, """# 02 - Offline Attack Evaluation (Gemma & Llama)
**Environment**: Kaggle GPU (T4 or RTX 6000) with **Internet DISABLED** (Air-gapped).

This notebook runs the jailbreak attack battery against your offline Gemma / Llama models with **no defense** initially.

---

### Setup Requirements:
Before running:
1. Turn **Internet OFF** in Kaggle notebook settings (right sidebar).
2. Attach your **`offline-attack-bundle`** dataset (from Notebook 1).
3. Attach your **Gemma** or **Llama** model dataset (e.g., from Kaggle Models or your custom dataset).
"""),

    (MD, "## 1 - Offline Pip Installation"),
    (CODE, """import os, sys, glob, pathlib, subprocess

# Locate wheels in attached Kaggle inputs or working directory
search_paths = [
    pathlib.Path("/kaggle/input/offline-attack-bundle/wheels"),
    pathlib.Path("/kaggle/input/offline-attack-bundle"),
    pathlib.Path("./offline_attack_bundle/wheels"),
    pathlib.Path("./wheels"),
]

wheels_dir = None
for p in search_paths:
    if p.exists() and list(p.glob("*.whl")):
        wheels_dir = p
        break

if not wheels_dir:
    # Check if a zip exists to extract
    zip_candidates = list(pathlib.Path("/kaggle/input").rglob("offline_attack_bundle.zip"))
    if zip_candidates:
        import zipfile
        out_dir = pathlib.Path("/kaggle/working/unpacked_bundle")
        with zipfile.ZipFile(zip_candidates[0], "r") as z:
            z.extractall(out_dir)
        wheels_dir = out_dir / "wheels"

if not wheels_dir or not list(wheels_dir.glob("*.whl")):
    raise RuntimeError("Could not find offline wheels directory. Please attach the offline-attack-bundle dataset.")

print(f"Installing wheels offline from: {wheels_dir}")
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "--no-index", f"--find-links={wheels_dir}",
    "transformers", "accelerate", "sentencepiece", "protobuf"
], check=True)
print("Offline dependencies installed successfully.")
"""),

    (MD, "## 2 - Load Repository Code & Environment Setup"),
    (CODE, """import shutil, pathlib, os, sys

# Locate repo root in /kaggle/input or local directories
src_repo = None
candidates = [
    *sorted([p.parent for p in pathlib.Path("/kaggle/input").rglob("run_eval.py")]),
    pathlib.Path("/kaggle/working/unpacked_bundle/repo"),
    pathlib.Path("./offline_attack_bundle/repo"),
    pathlib.Path("./repo"),
    pathlib.Path.cwd(),
]
for r in candidates:
    if (r / "run_eval.py").exists():
        src_repo = r.resolve()
        break

if not src_repo:
    raise RuntimeError("Could not find repository root containing run_eval.py.")

# On Kaggle, /kaggle/input is strictly read-only. Copy repo to /kaggle/working so it is fully writable!
dest_repo = pathlib.Path("/kaggle/working/repo")
if src_repo != dest_repo and not dest_repo.exists():
    shutil.copytree(src_repo, dest_repo)
    repo_root = dest_repo
else:
    repo_root = dest_repo if dest_repo.exists() else src_repo

print(f"Using repo at: {repo_root}")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

# Set offline environment variables
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
"""),

    (MD, "## 3 - Model & Attack Configuration"),
    (CODE, """# ==============================================================================
# CONFIGURE YOUR RUN HERE:
# ==============================================================================

# Path to your offline model weights (Gemma or Llama)
MODEL_PATH = "/kaggle/input/models/google/gemma-2/transformers/gemma-2-9b-it/2"

# Auto-check if the path exists, or auto-detect if located elsewhere in /kaggle/input
import pathlib
if not pathlib.Path(MODEL_PATH).exists():
    for candidate in pathlib.Path("/kaggle/input").rglob("*it*"):
        if candidate.is_dir() and ((candidate / "config.json").exists() or (candidate / "model.safetensors.index.json").exists()):
            MODEL_PATH = str(candidate)
            break

# Tag label for transcripts & output files
TAG = "gemma-9b-undefended"

# Attacks to run:
#   "all"                      -> all 17 active techniques
#   "prefix_injection,distractors,leetspeak" -> specific comma-separated list
ATTACKS = "all"

# Number of harmful prompts to test (1 to 50)
LIMIT = 50
# ==============================================================================

print(f"Target Model Path : {MODEL_PATH}")
print(f"Attacks           : {ATTACKS}")
print(f"Goal Limit        : {LIMIT}")
print(f"Run Tag           : {TAG}")
"""),

    (MD, "## 4 - Run the Attack Battery (No Defense)"),
    (CODE, """import time, os, sys, pathlib
import torch
import transformers as tf
import run_eval

print("Starting offline attack evaluation (defense=off, no-grade)...\\n")
t0 = time.time()

# 1. Direct model & writable logs configuration in memory:
from core.config import CONFIG
import core.models

CONFIG["models"]["target"]["name"] = str(MODEL_PATH)
CONFIG["models"]["target"]["revision"] = None
CONFIG["models"]["target"]["device"] = "auto"
CONFIG["models"]["target"]["max_memory"] = None

# Offline Safety: Helper model must NOT attempt to download remote weights (Qwen) while offline.
CONFIG["models"]["helper"]["backend"] = "fake"
CONFIG["models"]["judge"]["backend"] = "fake"
CONFIG["models"]["paraphraser"]["backend"] = "fake"
CONFIG["models"]["perplexity_scorer"]["backend"] = "fake"
core.models._load.cache_clear()
core.models.TransformersModelHandle._CACHE.clear()

# Route logs to /kaggle/working/logs (fully writable on Kaggle)
CONFIG["paths"]["logs_dir"] = "/kaggle/working/logs"
pathlib.Path("/kaggle/working/logs").mkdir(parents=True, exist_ok=True)
pathlib.Path("/kaggle/working/offload").mkdir(parents=True, exist_ok=True)

# Monkeypatch TransformersModelHandle._bundle to guarantee:
# - local_files_only=True is passed when loading local paths offline
# - revision is NOT passed for local directory paths
# - device_map="auto" with disk offload directory to handle 9B memory safely
def patched_bundle(self):
    key = (self.name, self._revision())
    cached = core.models.TransformersModelHandle._CACHE.get(key)
    if cached is not None:
        return cached

    _ver = tuple(int(x) for x in tf.__version__.split(".")[:2])
    _dtype_kw = "dtype" if _ver >= (4, 56) else "torch_dtype"

    is_local = os.path.isdir(str(self.name))
    load_kw = {}
    tok_kw = {}
    if is_local or self.spec.get("local_files_only"):
        load_kw["local_files_only"] = True
        tok_kw["local_files_only"] = True
    elif self._revision():
        load_kw["revision"] = self._revision()
        tok_kw["revision"] = self._revision()

    device = self.spec.get("device")
    if device and device != "auto":
        load_kw["device_map"] = {"": device}
    else:
        load_kw["device_map"] = "auto"
        load_kw["offload_folder"] = "/kaggle/working/offload"

    limits = self.spec.get("max_memory")
    if limits:
        load_kw["max_memory"] = {(int(k) if str(k).isdigit() else k): v
                                 for k, v in dict(limits).items()}

    qconf = self._quant_config()
    if qconf is not None:
        load_kw["quantization_config"] = qconf
    else:
        load_kw[_dtype_kw] = self._resolve_dtype()

    try:
        tokenizer = tf.AutoTokenizer.from_pretrained(self.name, **tok_kw)
        model = tf.AutoModelForCausalLM.from_pretrained(self.name, **load_kw)
        bundle = (model.eval(), tokenizer, "causal")
    except Exception:
        processor = tf.AutoProcessor.from_pretrained(self.name, **tok_kw)
        model = tf.AutoModelForImageTextToText.from_pretrained(self.name, **load_kw)
        bundle = (model.eval(), processor, "vlm")

    core.models.TransformersModelHandle._CACHE[key] = bundle
    return bundle

core.models.TransformersModelHandle._bundle = patched_bundle

# 2. Run the attacks
cmd_args = [
    "--attack", ATTACKS,
    "--defense", "off",
    "--no-grade",
    "--limit", str(LIMIT),
    "--tag", TAG,
]

exit_code = run_eval.main(cmd_args)

elapsed = (time.time() - t0) / 60
print(f"\\nRun finished with exit code {exit_code} in {elapsed:.1f} minutes.")
"""),

    (MD, "## 5 - Calculate Success Rates (ASR) & Inspect Responses"),
    (CODE, """import json, glob, os, pathlib
import pandas as pd
import report
from defense.layer4_response_classifier import heuristic_label, try_decode

# 1. Locate the newest transcript from /kaggle/working/logs
log_files = sorted(glob.glob("/kaggle/working/logs/*/transcript.jsonl"), key=os.path.getmtime)
if not log_files:
    print("No logs found in /kaggle/working/logs.")
else:
    transcript_path = pathlib.Path(log_files[-1])
    run_dir = transcript_path.parent
    print(f"Evaluating run: {run_dir.name}\\n")

    # 2. Grade each trial using refusal heuristics & ciphers decoding
    records = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    graded_records = []
    for r in records:
        if r.get("type") == "trial":
            resp = r.get("response", "")
            meta = r.get("metadata") or {}
            decode_scheme = meta.get("decode")
            
            # Decode if ciphertext attack (e.g. base64 / rot13)
            graded_text, was_decoded = try_decode(resp, decode_scheme)
            refusal = heuristic_label(graded_text)
            
            if refusal:
                label = "GOOD_BOT"      # Refused harmful goal -> attack failed
            elif not graded_text.strip():
                label = "UNCLEAR"       # Empty response
            else:
                label = "BAD_BOT"       # Complied with harmful request -> jailbreak succeeded
            
            r["label"] = label
            r["label_method"] = "heuristic" + (":decoded" if was_decoded else "")
        graded_records.append(r)

    # Save regraded transcript for report.py and downstream analyses
    regraded_path = run_dir / "transcript_regraded.jsonl"
    with open(regraded_path, "w", encoding="utf-8") as f:
        for r in graded_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\\n")

    # 3. Print per-attack Attack Success Rate (ASR) table
    print("=" * 75)
    print("ATTACK SUCCESS RATE (ASR) PER TECHNIQUE")
    print("=" * 75)
    tbl = report.print_asr(str(run_dir), regraded=True)
    
    # 4. Print adaptive attack summary
    print("\\n" + "=" * 75)
    print("ADAPTIVE ATTACK SUCCESS (Per-Goal Union)")
    print("=" * 75)
    try:
        report.print_adaptive(str(run_dir), regraded=True)
    except Exception as e:
        print("Adaptive calculation note:", e)

    # 5. Interactive preview table
    trials_df = pd.DataFrame([r for r in graded_records if r.get("type") == "trial"])
    if not trials_df.empty:
        summary_cols = ["attack", "label", "prompt", "response"]
        avail = [c for c in summary_cols if c in trials_df.columns]
        display(trials_df[avail].head(30))
"""),

    (MD, "## 6 - Package Results for Download"),
    (CODE, """import shutil

out_zip = pathlib.Path("/kaggle/working") / f"attack_results_{TAG}.zip"
logs_dir = pathlib.Path("/kaggle/working/logs") if pathlib.Path("/kaggle/working/logs").exists() else pathlib.Path("logs")
if logs_dir.exists():
    shutil.make_archive(str(out_zip.with_suffix("")), "zip", str(logs_dir))
    print(f"Results packaged at: {out_zip} ({out_zip.stat().st_size / 1024:.1f} KB)")
    print("Download this file from Kaggle's Output section to inspect the full responses!")
else:
    print("No logs directory to zip.")
"""),
]


def make_notebook(cells: list[tuple[str, str]], target_file: pathlib.Path) -> None:
    nb_cells = []
    for kind, src in cells:
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": src.splitlines(keepends=True),
        }
        if kind == CODE:
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)

    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {target_file} ({len(cells)} cells)")


def main() -> None:
    base = pathlib.Path(__file__).parent
    make_notebook(NB1_CELLS, base / "01_download_dependencies.ipynb")
    make_notebook(NB2_CELLS, base / "02_offline_attack_eval.ipynb")


if __name__ == "__main__":
    main()
