"""Smoke test -- everything registers and a dry run completes. Run: pytest -q

This is the guard that lets both halves work in parallel: if Member A's new technique
or Member B's layer change breaks the shared shape, this fails fast.
"""
from attacks import load_all
from attacks.base import CATEGORIES
from defense import build_pipeline


def test_every_attack_registers_with_a_known_category():
    attacks = load_all()
    assert {"passthrough", "base64", "prefix_injection", "refusal_suppression"} <= set(attacks)
    for name, atk in attacks.items():
        assert atk.category in CATEGORIES, f"{name}: bad category {atk.category!r}"


def test_pipeline_builds_in_order():
    pipe = build_pipeline(force_fake=True)
    assert [l.name for l in pipe.pre] == [
        "layer1_perplexity",
        "layer2_paraphrase",
        "layer3_system_hardening",
    ]
    assert [l.name for l in pipe.post] == ["layer4_response_classifier"]


def test_dry_run_end_to_end(tmp_path, monkeypatch):
    from core.config import CONFIG
    from run_eval import main

    monkeypatch.setitem(CONFIG["paths"], "logs_dir", str(tmp_path))
    assert main(["--dry-run", "--attack", "prefix_injection", "--limit", "1"]) == 0
    assert list(tmp_path.glob("*/transcript.jsonl")), "no transcript written"
