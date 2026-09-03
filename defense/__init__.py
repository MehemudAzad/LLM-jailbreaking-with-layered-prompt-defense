"""Assemble the defense stack from config.toml. FROZEN after Week 10 (order lives here).

Member B: to add or reorder a layer, edit `_ORDER` and add the matching
`[defense.<key>]` block in config.toml.
"""
from __future__ import annotations

from core.config import CONFIG
from defense.base import Action, DefenseContext, DefenseLayer, Stage, Verdict  # noqa: F401 (re-export)
from defense.pipeline import LayeredDefense
from defense.layer1_perplexity_filter import PerplexityFilter
from defense.layer2_paraphrase import ParaphraseDefense
from defense.layer3_system_hardening import SystemHardening
from defense.layer4_response_classifier import ResponseClassifier

_ORDER: list[tuple[str, type[DefenseLayer]]] = [
    ("layer1_perplexity", PerplexityFilter),
    ("layer2_paraphrase", ParaphraseDefense),
    ("layer3_system_hardening", SystemHardening),
    ("layer4_response_classifier", ResponseClassifier),
]


def build_pipeline(force_fake: bool = False) -> LayeredDefense:
    dcfg = CONFIG.get("defense", {})
    layers = [cls({**dcfg.get(key, {}), "_force_fake": force_fake}) for key, cls in _ORDER]
    return LayeredDefense(layers)
