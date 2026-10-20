"""Readable, seedable identifiers."""

import random


def new_id(prefix: str, rng: random.Random) -> str:
    """Return an id like `conv_3f9a1c2e`, drawn from `rng` so runs reproduce per seed."""
    return f"{prefix}_{rng.getrandbits(32):08x}"
