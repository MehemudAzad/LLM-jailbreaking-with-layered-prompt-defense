# logs/ — raw transcripts

One directory per run: `logs/<YYYYMMDD-HHMMSS>-<tag>-<short>/`

    meta.json         seed, pinned model names, python version
    transcript.jsonl  one JSON object per line

`transcript.jsonl` record types (`type` field):

| type      | when             | key fields |
|-----------|------------------|------------|
| `config`  | once, first      | seed, attack, defense, dry_run, n_goals, n_attacks |
| `trial`   | per goal × attack | goal_id, attack, category, blocked, blocked_by, prompt_sent, system, response, label, verdicts |
| `summary` | once, last       | trials, labels, asr, run_dir |

Rules:

- **Never edit a transcript after the fact.** If a run was wrong, re-run — the run-id
  keeps them unambiguous.
- Dev / dry-run transcripts are git-ignored by default (see `.gitignore`).
- For the **one full evaluation pass** (week 12) that both reports draw from, commit the
  run directory as evidence: `git add -f logs/<that-run-id>`.
