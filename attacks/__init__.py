"""Technique registry. FROZEN after Week 10.

Importing a technique module self-registers its class via `@register`. `load_all()`
imports every module in this package and returns {name: instance}.
"""
from __future__ import annotations

import importlib
import pkgutil

from attacks.base import AttackOutput, AttackTechnique, CATEGORIES  # noqa: F401  (re-exported)

_REGISTRY: dict[str, type[AttackTechnique]] = {}


def register(cls: type[AttackTechnique]) -> type[AttackTechnique]:
    if cls.name in _REGISTRY:
        raise ValueError(f"duplicate attack name: {cls.name!r}")
    if cls.category not in CATEGORIES:
        raise ValueError(f"{cls.name!r}: unknown category {cls.category!r} (pick from {CATEGORIES})")
    _REGISTRY[cls.name] = cls
    return cls


def load_all() -> dict[str, AttackTechnique]:
    for mod in pkgutil.iter_modules(__path__):
        if mod.name != "base":
            importlib.import_module(f"{__name__}.{mod.name}")
    return {name: cls() for name, cls in sorted(_REGISTRY.items())}


def get(name: str) -> AttackTechnique:
    if name not in _REGISTRY:
        load_all()
    return _REGISTRY[name]()
