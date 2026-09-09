#!/usr/bin/env python3
"""Read a finished run and render the tables/samples the reports need.

Deliberately a repo module, not notebook cells: notebook cells do NOT sync from git
(the live Kaggle notebook keeps its own copy), but imported modules do. Keep notebooks
thin -- import this and call it -- so fixes here reach every future run for free.

    import report
    report.print_asr('m4baseline')
    report.samples('m4baseline', exclude='GOOD_BOT', n=8)
    report.print_compare('m4baseline', 'm5defended')
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd

from core.config import resolve
from core.transcript import mirror_root


def find_run(tag_or_dir: str) -> pathlib.Path:
    """Newest run directory matching `tag_or_dir`, or the path itself if it is one.

    Searches logs/ and the mirror (on Kaggle logs/ is wiped when the repo is re-cloned,
    but /kaggle/working/artifacts survives).
    """
    direct = pathlib.Path(tag_or_dir)
    if (direct / "transcript.jsonl").exists():
        return direct

    roots = [resolve("logs")]
    mirror = mirror_root()
    if mirror is not None:
        roots.append(mirror)

    hits: list[pathlib.Path] = []
    for root in roots:
        if root.is_dir():
            hits += [p for p in root.glob(f"*{tag_or_dir}*") if (p / "transcript.jsonl").exists()]
    if not hits:
        raise FileNotFoundError(f"no run matching {tag_or_dir!r} under {[str(r) for r in roots]}")
    return sorted(hits, key=lambda p: p.name)[-1]


def load_run(tag_or_dir: str, regraded: bool = False) -> tuple[pathlib.Path, pd.DataFrame, dict]:
    """(run_dir, trials DataFrame, summary dict). `regraded=True` prefers regrade.py output."""
    run = find_run(tag_or_dir)
    name = "transcript_regraded.jsonl" if regraded else "transcript.jsonl"
    path = run / name
    if not path.exists():
        path = run / "transcript.jsonl"

    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    trials = pd.DataFrame([r for r in records if r.get("type") == "trial"])
    summary = next(
        (r for r in records if r.get("type", "").startswith("summary")),
        {},
    )
    return run, trials, summary


def asr_table(tag_or_dir: str, regraded: bool = False) -> pd.DataFrame:
    """Per-attack ASR. Blocked trials count as trials but never as BAD_BOT."""
    _, trials, _ = load_run(tag_or_dir, regraded)
    if trials.empty:
        return pd.DataFrame()

    rows = []
    for attack, grp in trials.groupby("attack"):
        n = len(grp)
        bad = int((grp.label == "BAD_BOT").sum())
        rows.append({
            "attack": attack,
            "category": grp.category.iloc[0] if "category" in grp else "",
            "n": n,
            "BAD_BOT": bad,
            "GOOD_BOT": int((grp.label == "GOOD_BOT").sum()),
            "UNCLEAR": int((grp.label == "UNCLEAR").sum()),
            "blocked": int(grp.blocked.sum()) if "blocked" in grp else 0,
            "ASR_%": round(100 * bad / n, 1) if n else None,
        })
    return pd.DataFrame(rows).set_index("attack").sort_values("ASR_%", ascending=False)


def print_asr(tag_or_dir: str, regraded: bool = False, title: str = "") -> pd.DataFrame:
    run, trials, summary = load_run(tag_or_dir, regraded)
    tbl = asr_table(tag_or_dir, regraded)
    n = len(trials)
    bad = int((trials.label == "BAD_BOT").sum()) if n else 0

    print(f"=== {title or tag_or_dir} ===")
    print(f"run: {run}")
    print(f"\n{tbl.to_string()}")
    print(f"\noverall ASR : {100 * bad / n:.1f}%   ({bad}/{n})")
    if n:
        print(f"label mix   : {trials.label.value_counts().to_dict()}")
        if "label_method" in trials:
            print(f"graded by   : {trials.label_method.value_counts(dropna=False).to_dict()}")
    return tbl


def samples(tag_or_dir: str, exclude: str | None = "GOOD_BOT", only: str | None = None,
            n: int = 8, width: int = 260, regraded: bool = False) -> None:
    """Print goal / reply / label triples for eyeballing the judge."""
    _, trials, _ = load_run(tag_or_dir, regraded)
    if trials.empty:
        print("no trials")
        return

    sel = trials
    if only:
        sel = sel[sel.label == only]
    elif exclude:
        sel = sel[sel.label != exclude]

    for _, row in sel.head(n).iterrows():
        meta = row.get("metadata") or {}
        print(f"[{row.attack}]  label={row.label}  ({row.get('label_method')})")
        print("  goal  :", str(meta.get("goal", ""))[:110])
        print("  reply :", (row.get("response") or "")[:width].replace("\n", " "))
        graded = meta.get("graded_text")
        if graded:
            print("  judged:", str(graded)[:width].replace("\n", " "), "   <- decoded")
        print()


def compare(baseline: str, defended: str, regraded: bool = False) -> pd.DataFrame:
    """Before/after ASR per attack -- the core result table for the report."""
    a = asr_table(baseline, regraded)[["category", "ASR_%"]].rename(columns={"ASR_%": "baseline_%"})
    b = asr_table(defended, regraded)[["ASR_%", "blocked"]].rename(columns={"ASR_%": "defended_%"})
    out = a.join(b, how="outer")
    out["drop_pp"] = (out["baseline_%"] - out["defended_%"]).round(1)
    return out.sort_values("baseline_%", ascending=False)


def print_compare(baseline: str, defended: str, regraded: bool = False) -> pd.DataFrame:
    tbl = compare(baseline, defended, regraded)
    print("=== baseline vs defended ASR ===\n")
    print(tbl.to_string())
    for col in ("baseline_%", "defended_%"):
        print(f"\nmean {col}: {tbl[col].mean():.1f}%")
    return tbl
