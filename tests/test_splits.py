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


def test_falls_back_to_stratified_split_when_too_few_joint_objects():
    """The real scenario hit in production: 0 joint objects (object_id/target_object_id_target
    namespace mismatch between the image and spectra sources). Joint-only val/test would be
    completely empty, which silently breaks early stopping (see assign_splits' docstring) — so
    below min_joint_objects, val/test must be drawn from all tiers instead, not left empty.
    """
    manifest = _synthetic_manifest(n_joint=0, n_single=60)
    cfg = OmegaConf.create({"splits": {"val": 0.1, "test": 0.1, "seed": 0}, "sanity": {"min_joint_objects": 500}})

    result = assign_splits(manifest, cfg)

    assert (result["split"] == "val").sum() > 0
    assert (result["split"] == "test").sum() > 0
    assert result.groupby("object_id")["split"].nunique().max() == 1


def test_uses_joint_only_when_above_threshold():
    manifest = _synthetic_manifest(n_joint=40, n_single=60)
    cfg = OmegaConf.create({"splits": {"val": 0.1, "test": 0.1, "seed": 0}, "sanity": {"min_joint_objects": 10}})

    result = assign_splits(manifest, cfg)

    non_train = result[result["split"] != "train"]
    assert (non_train["tier"] == "joint").all()


def _cfg_policy(policy: str, min_joint: int = 500):
    return OmegaConf.create(
        {"splits": {"val": 0.1, "test": 0.1, "seed": 0, "policy": policy},
         "sanity": {"min_joint_objects": min_joint}}
    )


def test_policy_defaults_to_auto_when_absent():
    """Configs predating splits.policy must behave exactly as before."""
    manifest = _synthetic_manifest(n_joint=40, n_single=60)
    result = assign_splits(manifest, _cfg())
    non_train = result[result["split"] != "train"]
    assert (non_train["tier"] == "joint").all()


def test_stratified_policy_ignores_a_plentiful_joint_tier():
    """The reason the policy exists: adding a source that pushes the joint count past
    min_joint_objects would otherwise silently flip val/test onto joint-tier objects only, making
    val_loss incomparable across runs for a reason nothing in the logs points at.
    """
    manifest = _synthetic_manifest(n_joint=600, n_single=600)
    result = assign_splits(manifest, _cfg_policy("stratified"))
    non_train = result[result["split"] != "train"]
    assert set(non_train["tier"]) == {"joint", "single"}
    assert result.groupby("object_id")["split"].nunique().max() == 1


def test_joint_only_policy_forces_joint_even_below_threshold():
    manifest = _synthetic_manifest(n_joint=40, n_single=60)
    result = assign_splits(manifest, _cfg_policy("joint_only"))
    non_train = result[result["split"] != "train"]
    assert (non_train["tier"] == "joint").all()


def test_joint_only_policy_raises_when_no_joint_objects_exist():
    """Empty val/test makes evaluate_loss return a fake 0.0 every epoch, which early stopping reads
    as 'no improvement' — silent truncation. Fail loudly instead."""
    manifest = _synthetic_manifest(n_joint=0, n_single=60)
    try:
        assign_splits(manifest, _cfg_policy("joint_only"))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "joint_only" in str(e)


def test_unknown_policy_raises():
    manifest = _synthetic_manifest()
    try:
        assign_splits(manifest, _cfg_policy("nonsense"))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "not recognised" in str(e)


def test_lightcurve_only_objects_are_eligible_under_stratified():
    """Transients are lightcurve-only, i.e. single-tier. They must be able to reach val/test."""
    rows = [{"object_id": f"lc_{i}", "tier": "single", "has_image": False,
             "has_spectra": False, "has_lightcurve": True} for i in range(100)]
    result = assign_splits(pd.DataFrame(rows), _cfg_policy("stratified"))
    assert (result["split"] == "val").sum() > 0
    assert (result["split"] == "test").sum() > 0
