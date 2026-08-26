"""cache_modality must never write a non-finite (NaN/Inf) embedding into the cache index as if
valid. Confirmed real failure mode: `_flux_to_array` in 02_cache_embeddings.py NaN-fills image
rows that come through as `None` (edge-clipped cutouts), on the assumption AION's encoder would
tolerate NaN input — it doesn't, and the NaN silently propagates all the way to a NaN training/
val loss with no visible cause. See data/cache.py's cache_modality docstring for the full chain.
"""
from __future__ import annotations

import pandas as pd
import torch
from omegaconf import OmegaConf

from captioner.data.cache import cache_modality


class _FakeEncoder:
    """Returns a NaN embedding for object_ids in `nan_ids`, a finite one otherwise."""

    def __init__(self, nan_ids: set[str]):
        self.nan_ids = nan_ids

    def encode(self, batch: dict) -> torch.Tensor:
        object_ids = batch["object_ids"]
        out = torch.ones(len(object_ids), 2, 3)
        for i, oid in enumerate(object_ids):
            if oid in self.nan_ids:
                out[i] = float("nan")
        return out


def _loader(object_ids: list[str]) -> dict:
    return {"object_ids": object_ids}


def _modality_cfg():
    return OmegaConf.create({"encoder": {"impl": "fake", "hf_path": "x/y", "revision": None, "kwargs": {}}})


def test_nan_embeddings_are_excluded_not_written(tmp_path):
    manifest = pd.DataFrame({"object_id": ["a", "b", "c"], "has_spectra": [True, True, True]})
    encoder = _FakeEncoder(nan_ids={"b"})

    target_dir, excluded_ids = cache_modality(
        "spectra", encoder, _modality_cfg(), manifest, _loader, tmp_path, shard=256
    )

    assert excluded_ids == ["b"]
    index_df = pd.read_parquet(target_dir / "index.parquet")
    assert set(index_df["object_id"]) == {"a", "c"}


def test_inf_embeddings_are_also_excluded(tmp_path):
    manifest = pd.DataFrame({"object_id": ["a", "b"], "has_spectra": [True, True]})

    class _InfEncoder:
        def encode(self, batch):
            out = torch.ones(len(batch["object_ids"]), 2, 3)
            out[0] = float("inf")
            return out

    target_dir, excluded_ids = cache_modality(
        "spectra", _InfEncoder(), _modality_cfg(), manifest, _loader, tmp_path, shard=256
    )

    assert excluded_ids == ["a"]
    index_df = pd.read_parquet(target_dir / "index.parquet")
    assert set(index_df["object_id"]) == {"b"}


def test_all_finite_batch_excludes_nothing(tmp_path):
    manifest = pd.DataFrame({"object_id": ["a", "b"], "has_spectra": [True, True]})
    encoder = _FakeEncoder(nan_ids=set())

    target_dir, excluded_ids = cache_modality(
        "spectra", encoder, _modality_cfg(), manifest, _loader, tmp_path, shard=256
    )

    assert excluded_ids == []
    index_df = pd.read_parquet(target_dir / "index.parquet")
    assert set(index_df["object_id"]) == {"a", "b"}


def test_excludes_span_multiple_shards(tmp_path):
    manifest = pd.DataFrame({"object_id": [f"o{i}" for i in range(5)], "has_spectra": [True] * 5})
    encoder = _FakeEncoder(nan_ids={"o1", "o4"})

    target_dir, excluded_ids = cache_modality(
        "spectra", encoder, _modality_cfg(), manifest, _loader, tmp_path, shard=2
    )

    assert set(excluded_ids) == {"o1", "o4"}
    index_df = pd.read_parquet(target_dir / "index.parquet")
    assert set(index_df["object_id"]) == {"o0", "o2", "o3"}
