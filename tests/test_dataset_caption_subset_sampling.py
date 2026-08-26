"""CaptionerDataset must only sample a target subset that both (a) has the underlying modality
data available and (b) actually has a caption written for that exact (object_id, subset) pair.
Confirmed real gap: spectra-tier captions are deliberately restricted to only the ~1,223
Gemini-covered objects (no rule-based fallback — an explicit scope decision), so most
has_spectra=True objects have embeddings but no caption for subset={"spectra"}. Sampling
`available` alone (the old behavior) would pick that subset anyway, land on an empty
caption_text, and — if enough such samples land in the same micro-batch — produce a batch whose
labels are entirely IGNORE_INDEX, which HF's CrossEntropyLoss turns into a NaN loss (0/0). That
NaN then poisons every trainable parameter's Adam state for good, since they're all shared
across the batch. This surfaced as an unexplained NaN train/val loss recurring after the
NaN-embedding cache fix (which fixed a different, unrelated cause) had already landed.
"""
from __future__ import annotations

import numpy as np

from captioner.data.dataset import CaptionerDataset


def _sample(available: frozenset, weights: list[tuple[frozenset, float]], captioned: set[frozenset], seed: int = 0):
    fake_self = type("Fake", (), {"subset_weights": weights})()
    rng = np.random.default_rng(seed)
    return CaptionerDataset._sample_target_subset(fake_self, available, rng, captioned)


def test_never_samples_a_subset_without_a_caption():
    # Data is available for both {"spectra"} and {"image", "spectra"}, but only the latter has
    # an actual caption — the old has_<modality>-only check would still offer {"spectra"}.
    weights = [(frozenset({"spectra"}), 0.5), (frozenset({"image", "spectra"}), 0.5)]
    captioned = {frozenset({"image", "spectra"})}

    for seed in range(50):
        shown = _sample(frozenset({"image", "spectra"}), weights, captioned, seed)
        assert shown != frozenset({"spectra"})
        assert shown in (frozenset(), frozenset({"image", "spectra"}))


def test_falls_back_to_empty_when_nothing_is_captioned():
    weights = [(frozenset({"spectra"}), 1.0)]
    shown = _sample(frozenset({"spectra"}), weights, captioned=set())
    assert shown == frozenset()


def test_samples_normally_when_everything_available_is_captioned():
    weights = [(frozenset({"image"}), 0.5), (frozenset({"spectra"}), 0.5)]
    captioned = {frozenset({"image"}), frozenset({"spectra"})}
    seen = {_sample(frozenset({"image", "spectra"}), weights, captioned, seed) for seed in range(50)}
    assert seen == {frozenset({"image"}), frozenset({"spectra"})}


def test_captioned_subsets_by_object_built_from_captions_table():
    import pandas as pd

    captions = pd.DataFrame(
        [
            {"object_id": "a", "subset": ["image"], "text": "an image caption"},
            {"object_id": "a", "subset": ["image", "spectra"], "text": "a joint caption"},
            {"object_id": "b", "subset": ["spectra"], "text": "a spectra caption"},
        ]
    )
    lookup: dict[str, set[frozenset]] = {}
    for _, row in captions.iterrows():
        lookup.setdefault(row["object_id"], set()).add(frozenset(row["subset"]))

    assert lookup["a"] == {frozenset({"image"}), frozenset({"image", "spectra"})}
    assert lookup["b"] == {frozenset({"spectra"})}
    assert "c" not in lookup
