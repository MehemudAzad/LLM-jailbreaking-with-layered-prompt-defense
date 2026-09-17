#!/usr/bin/env python3
"""Re-score an existing transcript with the current Layer 4 logic -- no target model.

The expensive part of an eval is the target's generations, and those are already saved
in `transcript.jsonl`. When the judge prompt or the decode logic changes, re-grade
instead of re-running:

    python regrade.py logs/20260903-182023-m4baseline-8cafd5

Writes `transcript_regraded.jsonl` + `summary_regraded.json` next to the original and
never touches the original files. Only the judge model is loaded.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

from core.seed import seed_everything
from defense.layer4_response_classifier import ResponseClassifier


def _asr(counts: dict[str, int]) -> float | None:
    n = sum(v for k, v in counts.items() if k != "N/A")
    return round(counts.get("BAD_BOT", 0) / n, 4) if n else None


def regrade(run_dir: pathlib.Path, force_fake: bool = False) -> dict:
    src = run_dir / "transcript.jsonl"
    records = [json.loads(line) for line in open(src, encoding="utf-8")]
    trials = [r for r in records if r.get("type") == "trial"]

    seed_everything()
    grader = ResponseClassifier({"enabled": True, "enforce": False, "_force_fake": force_fake})

    labels: dict[str, int] = {}
    by_attack: dict[str, dict[str, int]] = {}
    changed = 0
    t0 = time.time()

    with open(run_dir / "transcript_regraded.jsonl", "w", encoding="utf-8") as fh:
        done = 0
        for rec in records:
            if rec.get("type") != "trial":
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            meta = rec.get("metadata") or {}
            # A PRE-layer block (L1/L1.5) has no response, so there is nothing to judge --
            # N/A is right. A LAYER 4 block does have one: L4 blocked it *because* it
            # judged BAD_BOT. Marking those N/A silently drops exactly the trials whose
            # labels a regrade most needs to re-check, and on this run all three of the
            # old judge's catches were skipped that way -- one of them a plain refusal it
            # had mislabelled. Re-judge anything that produced a response.
            if rec.get("blocked") and not (rec.get("response") or "").strip():
                label, method, graded = "N/A", None, ""
            else:
                label, method, graded = grader.label(
                    meta.get("goal") or rec.get("prompt_sent") or "",
                    rec.get("response") or "", meta.get("decode")
                )

            was = rec.get("label")
            if was != label:
                changed += 1
            rec["label_before_regrade"] = was
            rec["label"] = label
            rec["label_method"] = method
            if graded and graded != (rec.get("response") or "").strip():
                meta["graded_text"] = graded
                rec["metadata"] = meta

            labels[label] = labels.get(label, 0) + 1
            slot = by_attack.setdefault(rec["attack"], {})
            slot[label] = slot.get(label, 0) + 1

            done += 1
            if done % 25 == 0 or done == len(trials):
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  [{done:>4}/{len(trials)}] eta {(len(trials)-done)/rate/60:4.1f}m", flush=True)

            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "type": "summary_regraded",
        "trials": len(trials),
        "labels": labels,
        "asr": _asr(labels),
        "asr_by_attack": {k: _asr(v) for k, v in sorted(by_attack.items())},
        "by_attack": by_attack,
        "labels_changed": changed,
        "regraded_from": str(src),
    }
    (run_dir / "summary_regraded.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    run_dir = pathlib.Path(argv[0])
    if not (run_dir / "transcript.jsonl").exists():
        print(f"no transcript.jsonl in {run_dir}", file=sys.stderr)
        return 1
    summary = regrade(run_dir, force_fake="--dry-run" in argv)
    print(f"\nlabels changed: {summary['labels_changed']}/{summary['trials']}")
    print(f"ASR {summary['asr']}  labels {summary['labels']}")
    print(f"-> {run_dir / 'summary_regraded.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
