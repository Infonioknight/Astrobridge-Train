"""CaptionerDataset must drop, at construction time, any object with no captioned subset at all
(data present but zero matching captions) — otherwise it sits in the split forever producing
empty-prompt/empty-caption training examples on every visit, every epoch, pure dead weight.
Object survives iff it has at least one of: a captioned image-only pair, a captioned spectra-only
pair, or a captioned joint pair — matches the same caption-existence check `_sample_target_subset`
uses (see test_dataset_caption_subset_sampling.py), just applied once up front instead of
re-discovered every __getitem__ call.

Builds a real on-disk cache (npy shard + index.parquet) so this exercises the actual
CaptionerDataset.__init__ path, not just the sampling helper in isolation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from captioner.data.cache import cache_dir_for
from captioner.encoders.registry import encoder_hash, encoder_spec
from captioner.data.dataset import CaptionerDataset


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None, return_tensors="pt"):
        import torch

        ids = [ord(c) % 50 for c in text] or [0]
        return {"input_ids": torch.tensor([ids])}


def _write_cache(cache_root: Path, modality: str, modality_cfg, object_ids: list[str]) -> None:
    spec = encoder_spec(modality_cfg)
    spec_hash = encoder_hash(spec)
    target_dir = cache_dir_for(cache_root, modality, spec_hash)
    target_dir.mkdir(parents=True, exist_ok=True)
    arr = np.ones((len(object_ids), 2, 3), dtype=np.float16)
    np.save(target_dir / "shard_00000.npy", arr)
    columns = {"object_id": pd.Series(dtype=str), "shard_file": pd.Series(dtype=str),
               "row_in_shard": pd.Series(dtype=int), "n_tokens": pd.Series(dtype=int), "out_dim": pd.Series(dtype=int)}
    index_df = pd.DataFrame(columns)
    if object_ids:
        index_df = pd.DataFrame(
            [
                {"object_id": oid, "shard_file": "shard_00000.npy", "row_in_shard": i, "n_tokens": 2, "out_dim": 3}
                for i, oid in enumerate(object_ids)
            ]
        )
    index_df.to_parquet(target_dir / "index.parquet", index=False)


@pytest.fixture
def modalities_cfg():
    return OmegaConf.create(
        {
            "modalities": {
                "image": {
                    "out_dim": 3, "max_tokens": 2,
                    "encoder": {"impl": "fake", "hf_path": "x/img", "revision": None, "kwargs": {}},
                },
                "spectra": {
                    "out_dim": 3, "max_tokens": 2,
                    "encoder": {"impl": "fake", "hf_path": "x/spec", "revision": None, "kwargs": {}},
                },
            },
            "dropout": {
                "subset_weights": [
                    {"subset": ["image"], "weight": 0.34},
                    {"subset": ["spectra"], "weight": 0.33},
                    {"subset": ["image", "spectra"], "weight": 0.33},
                ]
            },
        }
    )


def test_object_with_no_captioned_subset_is_dropped(tmp_path, modalities_cfg):
    manifest = pd.DataFrame(
        [
            # "keep": has spectra data and a real spectra caption.
            {"object_id": "keep", "has_image": False, "has_spectra": True, "split": "train"},
            # "drop": has spectra data but no caption for any subset it could show — dead weight.
            {"object_id": "drop", "has_image": False, "has_spectra": True, "split": "train"},
        ]
    )
    captions = pd.DataFrame([{"object_id": "keep", "subset": ["spectra"], "text": "a real caption"}])

    _write_cache(tmp_path, "image", modalities_cfg.modalities.image, [])
    _write_cache(tmp_path, "spectra", modalities_cfg.modalities.spectra, ["keep", "drop"])

    ds = CaptionerDataset(manifest, captions, modalities_cfg, tmp_path, "train", _FakeTokenizer(), "{modalities}")

    assert list(ds.manifest["object_id"]) == ["keep"]
    assert len(ds) == 1


def test_object_with_captioned_subset_survives_even_if_others_are_missing(tmp_path, modalities_cfg):
    manifest = pd.DataFrame(
        [
            {"object_id": "a", "has_image": True, "has_spectra": True, "split": "train"},
        ]
    )
    # Only the image-only subset has a caption — no spectra-only, no joint — but that's enough.
    captions = pd.DataFrame([{"object_id": "a", "subset": ["image"], "text": "an image caption"}])

    _write_cache(tmp_path, "image", modalities_cfg.modalities.image, ["a"])
    _write_cache(tmp_path, "spectra", modalities_cfg.modalities.spectra, ["a"])

    ds = CaptionerDataset(manifest, captions, modalities_cfg, tmp_path, "train", _FakeTokenizer(), "{modalities}")

    assert list(ds.manifest["object_id"]) == ["a"]


def test_all_objects_captioned_drops_nothing(tmp_path, modalities_cfg):
    manifest = pd.DataFrame(
        [
            {"object_id": "a", "has_image": False, "has_spectra": True, "split": "train"},
            {"object_id": "b", "has_image": False, "has_spectra": True, "split": "train"},
        ]
    )
    captions = pd.DataFrame(
        [
            {"object_id": "a", "subset": ["spectra"], "text": "caption a"},
            {"object_id": "b", "subset": ["spectra"], "text": "caption b"},
        ]
    )

    _write_cache(tmp_path, "image", modalities_cfg.modalities.image, [])
    _write_cache(tmp_path, "spectra", modalities_cfg.modalities.spectra, ["a", "b"])

    ds = CaptionerDataset(manifest, captions, modalities_cfg, tmp_path, "train", _FakeTokenizer(), "{modalities}")

    assert set(ds.manifest["object_id"]) == {"a", "b"}
