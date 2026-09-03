#!/usr/bin/env python3
"""Freeze the graded harmful-behaviour set from AdvBench (Zou et al. 2023).

Deterministic: same seed -> same 50 rows. Regenerate any time:

    python datasets/build_harmful.py            # default N, seed from below
    python datasets/build_harmful.py 120        # a different subset size
    python datasets/build_harmful.py 520        # the full benchmark

Source, in order of preference (identical 520 (goal, target) rows, MIT-licensed):
  1. datasets/advBench/**/*.parquet   -- the walledai/AdvBench mirror, if downloaded
  2. https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv

Writes datasets/harmful_behaviors.jsonl (git-ignored -- public repo).
"""
from __future__ import annotations

import csv
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 1337  # keep in sync with config.toml  [seed]
DEFAULT_N = 50
CSV_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
OUT = HERE / "harmful_behaviors.jsonl"


def _from_parquet() -> list[dict] | None:
    hits = sorted((HERE / "advBench").glob("**/*.parquet"))
    if not hits:
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(hits[0])
    except ImportError:
        print("(local parquet found, but pandas/pyarrow unavailable -- using the CSV instead)")
        return None
    goal_col = "prompt" if "prompt" in df.columns else "goal"
    return [{"goal": str(r[goal_col]), "target": str(r["target"])} for _, r in df.iterrows()]


def _from_csv() -> list[dict]:
    with urllib.request.urlopen(CSV_URL, timeout=30) as resp:
        rows = list(csv.DictReader(io.StringIO(resp.read().decode("utf-8"))))
    return [{"goal": r["goal"], "target": r["target"]} for r in rows]


def main(argv: list[str]) -> int:
    n = int(argv[0]) if argv else DEFAULT_N

    rows = _from_parquet()
    src = "datasets/advBench parquet"
    if rows is None:
        rows = _from_csv()
        src = "llm-attacks CSV"
    print(f"loaded {len(rows)} behaviours from {src}")

    idx = sorted(random.Random(SEED).sample(range(len(rows)), min(n, len(rows))))
    picked = []
    for out_i, j in enumerate(idx, start=1):
        target = rows[j]["target"].strip()
        prefix = " ".join(target.split()[:4]) or "Sure, here is"  # generic affirmative opener
        picked.append({
            "id": f"hb_{out_i:04d}",
            "category": "advbench",
            "goal": rows[j]["goal"].strip(),
            "target_prefix": prefix,
            "source": f"AdvBench harmful_behaviors #{j} (Zou et al. 2023)",
        })

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(f"// FROZEN -- {len(picked)}/{len(rows)} AdvBench behaviours, seed {SEED}. "
                 f"Regenerate: python datasets/build_harmful.py {len(picked)}\n")
        for r in picked:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {OUT.relative_to(HERE.parent)}  ({len(picked)} rows)")
    print(f"-> set config.toml  [datasets] harmful_count = {len(picked)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
