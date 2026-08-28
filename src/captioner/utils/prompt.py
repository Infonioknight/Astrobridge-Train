"""Shared prompt-text formatting — used by both data/dataset.py (training/eval, reading from
cache) and inference.py (live, uncached single-object inference), so the prompt wording an
object was trained against is guaranteed identical to what inference constructs at serve time.
"""
from __future__ import annotations

# Human-readable phrase per modality, so prompts read naturally as a modality is added. A modality
# with no entry falls back to "a <name>", which is right for most names but wrong for "image" (needs
# "an") and clumsy for compound names like "lightcurve".
DISPLAY_NAMES = {
    "image": "an image",
    "spectra": "a spectrum",
    "lightcurve": "a light curve",
}


def _display(name: str) -> str:
    return DISPLAY_NAMES.get(name, f"a {name}")


def human_readable_subset(subset: frozenset[str]) -> str:
    names = sorted(subset)
    if not names:
        return ""
    return " and ".join(_display(n) for n in names)
