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


def attribution(tag_or_dir: str, regraded: bool = False) -> pd.DataFrame:
    """Which layer stopped each attack -- attack x blocking-layer counts.

    The defended pass's headline number ("ASR fell from X% to Y%") says nothing about
    *why*. This reads the per-trial `blocked_by` field the pipeline records and credits
    each block to the layer that actually made it, so the report can say e.g. "the whole
    base64 family is stopped at L1.5, persona attacks survive to L4". `reached_target`
    counts trials no PRE layer blocked (L4 judges those after the model answered).
    """
    _, trials, _ = load_run(tag_or_dir, regraded)
    if trials.empty:
        return pd.DataFrame()

    by = trials.get("blocked_by")
    if by is None:
        return pd.DataFrame()
    who = by.fillna("reached_target").replace({None: "reached_target", "": "reached_target"})

    tbl = pd.crosstab(trials["attack"], who)
    if "reached_target" not in tbl:
        tbl["reached_target"] = 0
    # order columns by pipeline position, then the pass-through bucket last
    order = [c for c in ("layer1_perplexity", "layer1_5_structural", "layer2_paraphrase",
                         "layer3_system_hardening", "layer4_response_classifier") if c in tbl]
    tbl = tbl[order + [c for c in tbl.columns if c not in order and c != "reached_target"]
              + ["reached_target"]]
    tbl["n"] = tbl.sum(axis=1)
    return tbl.sort_values("reached_target")


def print_attribution(tag_or_dir: str, regraded: bool = False, title: str = "") -> pd.DataFrame:
    tbl = attribution(tag_or_dir, regraded)
    print(f"=== {title or 'per-layer attribution'} ===")
    if tbl.empty:
        print("no blocked_by data in this run (was it run with --defense off?)")
        return tbl
    print(f"\n{tbl.to_string()}")
    totals = tbl.drop(columns=["n"]).sum()
    grand = int(totals.sum())
    print("\nshare of all trials stopped, by layer:")
    for layer, cnt in totals.sort_values(ascending=False).items():
        if cnt:
            print(f"  {layer:28} {int(cnt):4}  ({100 * cnt / grand:4.1f}%)")
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


def regrade_diff(tag_or_dir: str) -> pd.DataFrame:
    """Per-attack ASR before vs after a regrade (needs transcript_regraded.jsonl)."""
    run = find_run(tag_or_dir)
    path = run / "transcript_regraded.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run regrade.py on this run first")

    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    trials = pd.DataFrame([r for r in records if r.get("type") == "trial"])
    if trials.empty:
        return pd.DataFrame()

    rows = []
    for attack, grp in trials.groupby("attack"):
        n = len(grp)
        rows.append({
            "attack": attack,
            "n": n,
            "before_%": round(100 * (grp.label_before_regrade == "BAD_BOT").sum() / n, 1),
            "after_%": round(100 * (grp.label == "BAD_BOT").sum() / n, 1),
            "changed": int((grp.label_before_regrade != grp.label).sum()),
        })
    out = pd.DataFrame(rows).set_index("attack")
    out["delta_pp"] = (out["after_%"] - out["before_%"]).round(1)
    return out.sort_values("before_%", ascending=False)


def print_regrade_diff(tag_or_dir: str) -> pd.DataFrame:
    tbl = regrade_diff(tag_or_dir)
    _, trials, _ = load_run(tag_or_dir, regraded=True)
    n = len(trials)
    before = int((trials.label_before_regrade == "BAD_BOT").sum())
    after = int((trials.label == "BAD_BOT").sum())

    print("=== regrade: ASR before vs after ===\n")
    print(tbl.to_string())
    print(f"\noverall ASR : {100*before/n:.1f}%  ->  {100*after/n:.1f}%   ({before} -> {after} BAD_BOT of {n})")
    print(f"labels changed: {int((trials.label_before_regrade != trials.label).sum())}/{n}")
    print(f"label mix after: {trials.label.value_counts().to_dict()}")
    return tbl


def changed_samples(tag_or_dir: str, n: int = 8, width: int = 220) -> None:
    """Show trials whose label moved during a regrade -- the ones worth eyeballing."""
    _, trials, _ = load_run(tag_or_dir, regraded=True)
    moved = trials[trials.label_before_regrade != trials.label]
    print(f"{len(moved)} labels changed; showing {min(n, len(moved))}\n")
    for _, row in moved.head(n).iterrows():
        meta = row.get("metadata") or {}
        print(f"[{row.attack}]  {row.label_before_regrade} -> {row.label}  ({row.get('label_method')})")
        print("  goal  :", str(meta.get("goal", ""))[:100])
        print("  reply :", (row.get("response") or "")[:width].replace("\n", " "))
        graded = meta.get("graded_text")
        if graded:
            print("  judged:", str(graded)[:width].replace("\n", " "), "   <- decoded")
        print()


def compare(run_a: str, run_b: str, label_a: str = "baseline", label_b: str = "defended",
            regraded: bool = False) -> pd.DataFrame:
    """Per-attack ASR for two runs side by side.

    Works for baseline-vs-defended and equally for 3B-vs-7B -- just pass labels.
    """
    ca, cb = f"{label_a}_%", f"{label_b}_%"
    a = asr_table(run_a, regraded)[["category", "ASR_%"]].rename(columns={"ASR_%": ca})
    b = asr_table(run_b, regraded)[["ASR_%", "blocked"]].rename(columns={"ASR_%": cb})
    out = a.join(b, how="outer")
    out["delta_pp"] = (out[cb] - out[ca]).round(1)
    return out.sort_values(cb, ascending=False)


def print_compare(run_a: str, run_b: str, label_a: str = "baseline", label_b: str = "defended",
                  regraded: bool = False) -> pd.DataFrame:
    tbl = compare(run_a, run_b, label_a, label_b, regraded)
    ca, cb = f"{label_a}_%", f"{label_b}_%"
    print(f"=== {label_a} vs {label_b}: ASR per attack ===\n")
    print(tbl.to_string())
    print(f"\nmean {label_a}: {tbl[ca].mean():.1f}%     mean {label_b}: {tbl[cb].mean():.1f}%")
    return tbl
