"""Stable deterministic seeding helpers."""

from __future__ import annotations

import hashlib


def stable_seed(base_seed: int, *parts: object, bits: int = 31) -> int:
    """Derive a deterministic integer seed from a base seed and arbitrary parts."""
    payload = "|".join([str(int(base_seed))] + [str(part) for part in parts]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    mask = (1 << int(bits)) - 1
    return int(value & mask)
