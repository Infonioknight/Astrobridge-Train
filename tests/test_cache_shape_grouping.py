"""data/cache.py's shard grouping. Encoders take one stacked tensor per call, so a batch whose
members disagree on cutout shape cannot be built — real case: gapatron/astrobridge-image-captions
ships legacy-south at 160x160 and legacy-north at 152x152 in one dataset, and sharding in plain
manifest order eventually straddles that boundary and dies inside _image_batch_loader.
"""
from __future__ import annotations

from captioner.data.cache import _shard_object_ids


def test_no_group_key_is_plain_contiguous_sharding():
    ids = [f"o{i}" for i in range(7)]
    assert _shard_object_ids(ids, 3, None) == [["o0", "o1", "o2"], ["o3", "o4", "o5"], ["o6"]]


def test_shards_never_mix_two_group_keys():
    ids = ["s1", "n1", "s2", "n2", "s3"]
    groups = {"s1": (3, 160, 160), "s2": (3, 160, 160), "s3": (3, 160, 160),
              "n1": (3, 152, 152), "n2": (3, 152, 152)}

    batches = _shard_object_ids(ids, 4, groups)

    assert all(len({groups[o] for o in batch}) == 1 for batch in batches)
    assert sorted(o for batch in batches for o in batch) == sorted(ids)


def test_groups_larger_than_the_shard_size_still_split():
    ids = [f"s{i}" for i in range(5)] + [f"n{i}" for i in range(3)]
    groups = {**{f"s{i}": "south" for i in range(5)}, **{f"n{i}": "north" for i in range(3)}}

    batches = _shard_object_ids(ids, 2, groups)

    assert [len(b) for b in batches] == [2, 2, 1, 2, 1]
    assert all(len({groups[o] for o in batch}) == 1 for batch in batches)


def test_ordering_is_deterministic_for_a_given_manifest():
    """Re-running the cache must reproduce the same shards — index.parquet records which shard an
    object landed in, so a reshuffle between runs would silently invalidate an existing index.
    """
    ids = ["s1", "n1", "s2", "n2"]
    groups = {"s1": "south", "s2": "south", "n1": "north", "n2": "north"}
    assert _shard_object_ids(ids, 2, groups) == _shard_object_ids(ids, 2, groups)
    assert _shard_object_ids(ids, 2, groups) == [["s1", "s2"], ["n1", "n2"]]


def test_every_object_appears_exactly_once():
    ids = [f"o{i}" for i in range(20)]
    groups = {o: i % 3 for i, o in enumerate(ids)}
    flat = [o for batch in _shard_object_ids(ids, 4, groups) for o in batch]
    assert sorted(flat) == sorted(ids)
    assert len(flat) == len(set(flat))
