"""Loads config.toml -- the single source of truth for the reproducible demo.

`config.local.toml`, if present, is deep-merged on top (per-machine overrides:
device, backend, cache paths). It is git-ignored.
"""
from __future__ import annotations

import copy
import functools
from pathlib import Path
from typing import Any

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore  (pip install tomli)

REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT = REPO_ROOT / "config.toml"
_LOCAL = REPO_ROOT / "config.local.toml"


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


@functools.lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else _DEFAULT
    with open(cfg_path, "rb") as fh:
        cfg = tomllib.load(fh)
    if path is None and _LOCAL.exists():
        with open(_LOCAL, "rb") as fh:
            cfg = _deep_merge(cfg, tomllib.load(fh))
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def resolve(rel: str | Path) -> Path:
    """Resolve a repo-relative path (as written in config.toml) against the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


CONFIG: dict[str, Any] = load_config()
