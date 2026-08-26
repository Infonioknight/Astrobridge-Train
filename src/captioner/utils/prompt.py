"""Shared prompt-text formatting — used by both data/dataset.py (training/eval, reading from
cache) and inference.py (live, uncached single-object inference), so the prompt wording an
object was trained against is guaranteed identical to what inference constructs at serve time.
"""
from __future__ import annotations


def human_readable_subset(subset: frozenset[str]) -> str:
    names = sorted(subset)
    if len(names) == 1:
        return f"a {names[0]}" if names[0] != "image" else "an image"
    return " and ".join(names)
