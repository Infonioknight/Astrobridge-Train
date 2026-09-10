"""build_manifest / assign_splits — IMAGE-ONLY BRANCH. One modality: the image identity table
IS the manifest, every row is has_image=True / tier="single", and val/test are a seeded draw
over all objects (no join, no upstream split column, no joint tier).
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from captioner.data.manifest import assign_splits, build_manifest, write_manifest


def _cfg(**over):
    base = {
        "sources": {"image": {"hf_path": "y"}, "image_captions": {"hf_path": "z"}},
        "join": {"key": "_healpix_29", "fallback_radius_arcsec": 1.0},
        "sanity": {"min_joint_objects": 500, "min_per_subset": 50},
        "splits": {"val": 0.1, "test": 0.1, "seed": 0, "policy": "stratified", "honor_upstream": True},
        "manifest": {"out_dir": "/tmp/_m", "parquet": "/tmp/_m/manifest.parquet", "stats": "/tmp/_m/stats.json"},
    }
    base.update(over)
    return OmegaConf.create(base)


def _image_df(n=100):
    return pd.DataFrame(
        {
            "object_id": [f"obj{i}" for i in range(n)],
            "object_id_legacy": [f"{i:04d}p000-0000" for i in range(n)],
            "ra": np.linspace(0, 359, n),
            "dec": np.linspace(-30, 30, n),
            "has_image": [True] * n,
            "image_legacy": [{"flux": [1.0]}] * n,  # heavy — must be dropped
        }
    )


def test_image_table_becomes_the_manifest_single_tier():
    with patch("captioner.data.manifest._load_image_table", return_value=_image_df(10).assign()):
        manifest, stats = build_manifest(_cfg())
    assert len(manifest) == 10
    assert manifest["has_image"].all()
    assert (manifest["tier"] == "single").all()
    assert stats["modalities"] == ["image"]
    assert stats["n_joint"] == 0


def test_heavy_pixel_column_is_dropped():
    with patch("captioner.data.manifest._load_image_table", return_value=_image_df(5)):
        manifest, _ = build_manifest(_cfg())
    assert "image_legacy" not in manifest.columns
    assert "object_id_legacy" in manifest.columns  # scalar metadata kept


def test_duplicate_object_id_raises():
    df = _image_df(4)
    df.loc[3, "object_id"] = "obj0"
    with patch("captioner.data.manifest._load_image_table", return_value=df):
        with pytest.raises(ValueError, match="duplicated object_id"):
            build_manifest(_cfg())


def test_assign_splits_is_seeded_and_disjoint():
    with patch("captioner.data.manifest._load_image_table", return_value=_image_df(100)):
        manifest, _ = build_manifest(_cfg())

    a = assign_splits(manifest, _cfg())
    b = assign_splits(manifest, _cfg())
    assert a.set_index("object_id")["split"].to_dict() == b.set_index("object_id")["split"].to_dict()

    c = assign_splits(manifest, _cfg(splits={"val": 0.1, "test": 0.1, "seed": 7, "policy": "stratified", "honor_upstream": True}))
    assert a["split"].tolist() != c["split"].tolist()

    counts = a["split"].value_counts().to_dict()
    assert counts["val"] == 10 and counts["test"] == 10 and counts["train"] == 80
    train_ids = set(a.loc[a["split"] == "train", "object_id"])
    other_ids = set(a.loc[a["split"] != "train", "object_id"])
    assert train_ids.isdisjoint(other_ids)


def test_write_manifest_raises_on_empty_split(tmp_path):
    cfg = _cfg(manifest={"out_dir": str(tmp_path), "parquet": str(tmp_path / "m.parquet"), "stats": str(tmp_path / "s.json")})
    with patch("captioner.data.manifest._load_image_table", return_value=_image_df(3)):  # 10% of 3 rounds to 0
        with pytest.raises(ValueError, match="empty"):
            write_manifest(cfg)
