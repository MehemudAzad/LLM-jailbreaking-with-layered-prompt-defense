"""One place to seed every RNG. Call `seed_everything()` at the top of any entrypoint."""
from __future__ import annotations

import os
import random

from core.config import CONFIG


def seed_everything(seed: int | None = None) -> int:
    """Seed random / numpy / torch. Returns the seed actually used."""
    if seed is None:
        seed = int(CONFIG.get("seed", 0))

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ModuleNotFoundError:
        pass

    return seed
