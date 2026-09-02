from __future__ import annotations

import pandas as pd
from omegaconf import OmegaConf

from captioner.data.manifest import assign_splits


def _cfg(**overrides):
    splits = {"policy": "stratified", "honor_upstream": True, "val": 0.1, "test": 0.1, "seed": 0}
    splits.update(overrides)
    return OmegaConf.create({"splits": splits, "sanity": {"min_joint_objects": 500}})


def _manifest(n_upstream_train=100, n_upstream_test=20, n_unlabelled=100, tier="single"):
    rows = []
    for i in range(n_upstream_train):
        rows.append({"object_id": f"up_train_{i}", "tier": tier, "split_upstream": "train"})
    for i in range(n_upstream_test):
        rows.append({"object_id": f"up_test_{i}", "tier": tier, "split_upstream": "test"})
    for i in range(n_unlabelled):
        rows.append({"object_id": f"unlabelled_{i}", "tier": tier, "split_upstream": None})
    return pd.DataFrame(rows)


def test_every_upstream_test_object_lands_in_test():
    out = assign_splits(_manifest(), _cfg()).set_index("object_id")
    upstream_test = [f"up_test_{i}" for i in range(20)]
    assert (out.loc[upstream_test, "split"] == "test").all()


def test_no_upstream_test_object_is_ever_trained_on():
    out = assign_splits(_manifest(), _cfg())
    leaked = out[(out["split_upstream"] == "test") & (out["split"] == "train")]
    assert len(leaked) == 0


def test_val_is_carved_out_of_the_upstream_train_pool_only():
    out = assign_splits(_manifest(), _cfg()).set_index("object_id")
    val = out[out["split"] == "val"]
    assert len(val) > 0
    assert not any(oid.startswith("up_test_") for oid in val.index)
    assert sum(oid.startswith("up_train_") for oid in val.index) == 10


def test_unlabelled_objects_still_get_a_seeded_draw():
    out = assign_splits(_manifest(), _cfg())
    unlabelled = out[out["object_id"].str.startswith("unlabelled_")]
    assert set(unlabelled["split"]) == {"train", "val", "test"}
    assert (unlabelled["split"] == "val").sum() == 10
    assert (unlabelled["split"] == "test").sum() == 10


def test_upstream_labels_beat_the_seed():
    a = assign_splits(_manifest(), _cfg(seed=0)).set_index("object_id")["split"]
    b = assign_splits(_manifest(), _cfg(seed=99)).set_index("object_id")["split"]
    upstream_test = [f"up_test_{i}" for i in range(20)]
    assert (a.loc[upstream_test] == b.loc[upstream_test]).all()
    unlabelled = [f"unlabelled_{i}" for i in range(100)]
    assert not (a.loc[unlabelled] == b.loc[unlabelled]).all()


def test_honor_upstream_false_restores_the_fully_random_draw():
    out = assign_splits(_manifest(), _cfg(honor_upstream=False))
    upstream_test = out[out["split_upstream"] == "test"]
    assert set(upstream_test["split"]) != {"test"}
    assert (out["split"] == "val").sum() == 22


def test_splits_are_stratified_across_tiers():
    joint = _manifest(n_upstream_train=50, n_upstream_test=10, n_unlabelled=0, tier="joint")
    single = _manifest(n_upstream_train=50, n_upstream_test=10, n_unlabelled=0, tier="single")
    single["object_id"] = single["object_id"] + "_s"
    out = assign_splits(pd.concat([joint, single], ignore_index=True), _cfg())
    val = out[out["split"] == "val"]
    assert (val["tier"] == "joint").sum() == 5
    assert (val["tier"] == "single").sum() == 5
