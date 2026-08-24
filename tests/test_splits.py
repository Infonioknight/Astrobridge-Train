"""§1 splits / §10 pitfall 5: splits are object-level. The same object_id must never appear in
two splits, and val/test must be drawn only from joint-tier objects.
"""
from __future__ import annotations

import pandas as pd
from omegaconf import OmegaConf

from captioner.data.manifest import assign_splits


def _synthetic_manifest(n_joint: int = 40, n_single: int = 60) -> pd.DataFrame:
    rows = []
    for i in range(n_joint):
        rows.append({"object_id": f"joint_{i}", "tier": "joint", "has_image": True, "has_spectra": True})
    for i in range(n_single):
        rows.append({"object_id": f"single_{i}", "tier": "single", "has_image": (i % 2 == 0), "has_spectra": (i % 2 == 1)})
    return pd.DataFrame(rows)


def _cfg():
    return OmegaConf.create({"splits": {"val": 0.1, "test": 0.1, "seed": 0}})


def test_no_object_id_appears_in_two_splits():
    manifest = _synthetic_manifest()
    result = assign_splits(manifest, _cfg())
    assert result.groupby("object_id")["split"].nunique().max() == 1


def test_val_and_test_are_joint_only():
    manifest = _synthetic_manifest()
    result = assign_splits(manifest, _cfg())
    non_train = result[result["split"] != "train"]
    assert (non_train["tier"] == "joint").all()


def test_unpaired_objects_go_to_train_only():
    manifest = _synthetic_manifest()
    result = assign_splits(manifest, _cfg())
    single_rows = result[result["tier"] == "single"]
    assert (single_rows["split"] == "train").all()
