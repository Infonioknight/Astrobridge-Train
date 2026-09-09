"""eval/datasets/image_galaxy10.py's stratified_sample — every random decision must be
reproducible given the same seed, and every one of the 10 classes must be represented.
"""
from __future__ import annotations

import pandas as pd
import pytest

from eval.datasets.image_galaxy10 import GALAXY10_LABELS, stratified_sample

# Mirrors the real, confirmed per-class counts on the actual test split.
_REAL_COUNTS = [49, 87, 148, 69, 8, 86, 90, 90, 46, 123]


def _synthetic_table() -> pd.DataFrame:
    labels, uids = [], []
    uid = 0
    for label, n in zip(GALAXY10_LABELS, _REAL_COUNTS):
        for _ in range(n):
            labels.append(label)
            uids.append(uid)
            uid += 1
    return pd.DataFrame({"label_name": labels, "uid": uids})


def test_same_seed_is_fully_reproducible():
    table = _synthetic_table()
    s1 = stratified_sample(table, 150, seed=42, min_per_class=2)
    s2 = stratified_sample(table, 150, seed=42, min_per_class=2)
    assert s1["uid"].tolist() == s2["uid"].tolist()


def test_different_seeds_draw_different_objects():
    table = _synthetic_table()
    s1 = stratified_sample(table, 150, seed=42, min_per_class=2)
    s2 = stratified_sample(table, 150, seed=7, min_per_class=2)
    assert s1["uid"].tolist() != s2["uid"].tolist()


def test_all_ten_classes_represented():
    table = _synthetic_table()
    sample = stratified_sample(table, 150, seed=0, min_per_class=2)
    assert set(sample["label_name"]) == set(GALAXY10_LABELS)


def test_sample_size_is_exact():
    table = _synthetic_table()
    sample = stratified_sample(table, 150, seed=0, min_per_class=2)
    assert len(sample) == 150


def test_rarest_class_gets_at_least_min_per_class():
    table = _synthetic_table()
    sample = stratified_sample(table, 150, seed=0, min_per_class=2)
    counts = sample["label_name"].value_counts()
    assert counts["Cigar Shaped Smooth Galaxies"] >= 2


def test_min_per_class_capped_at_available_not_padded():
    table = _synthetic_table()
    # Cigar Shaped Smooth Galaxies only has 8 real objects — asking for more than that as a floor
    # must not duplicate rows to hit it. min_per_class=10 kept below n_total=150's min-feasible
    # threshold (108) so this exercises the cap itself, not the separate infeasibility check.
    sample = stratified_sample(table, 150, seed=0, min_per_class=10)
    assert sample["label_name"].value_counts()["Cigar Shaped Smooth Galaxies"] <= 8
    assert sample["uid"].duplicated().sum() == 0


def test_n_total_exceeding_available_raises():
    table = _synthetic_table()
    with pytest.raises(ValueError, match="exceeds"):
        stratified_sample(table, 10_000, seed=0)


def test_n_total_below_min_feasible_raises_rather_than_silently_growing():
    table = _synthetic_table()
    with pytest.raises(ValueError, match="is below"):
        stratified_sample(table, 15, seed=0, min_per_class=2)


def test_no_duplicate_rows_within_a_sample():
    table = _synthetic_table()
    sample = stratified_sample(table, 150, seed=0, min_per_class=2)
    assert sample["uid"].duplicated().sum() == 0
