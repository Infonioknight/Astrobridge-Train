"""eval/datasets/gz_decals_votes.py's crossmatch_to_sample — pure logic over synthetic
coordinates, no network. The real caught bug here (bool `.mask` columns can't hold `NaN`) is
exactly what test_unmatched_row_mask_columns_become_true_not_nan guards against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eval.datasets.gz_decals_votes import _MASK_COLUMNS, _VOTE_COLUMNS, crossmatch_to_sample


def _synthetic_votes() -> pd.DataFrame:
    return pd.DataFrame({
        "ra": [10.0, 20.0, 30.0],
        "dec": [0.0, 0.0, 0.0],
        "iauname": ["a", "b", "c"],
        **{c: [0.5, 0.6, 0.7] for c in _VOTE_COLUMNS},
        **{c: [False, False, False] for c in _MASK_COLUMNS},
    })


def test_close_match_within_radius_joins_correctly():
    sample = pd.DataFrame({"ra": [10.0001], "dec": [0.0]})
    result = crossmatch_to_sample(sample, _synthetic_votes(), radius_arcsec=1.0)
    assert result["_crossmatched"].iloc[0]
    assert result["iauname"].iloc[0] == "a"
    assert result[_VOTE_COLUMNS[0]].iloc[0] == 0.5


def test_far_object_does_not_match():
    sample = pd.DataFrame({"ra": [99.0], "dec": [0.0]})
    result = crossmatch_to_sample(sample, _synthetic_votes(), radius_arcsec=1.0)
    assert not result["_crossmatched"].iloc[0]


def test_unmatched_row_mask_columns_become_true_not_nan():
    sample = pd.DataFrame({"ra": [99.0], "dec": [0.0]})
    result = crossmatch_to_sample(sample, _synthetic_votes(), radius_arcsec=1.0)
    for col in _MASK_COLUMNS:
        assert result[col].iloc[0] is True or result[col].iloc[0] == True  # noqa: E712


def test_unmatched_row_vote_columns_become_nan():
    sample = pd.DataFrame({"ra": [99.0], "dec": [0.0]})
    result = crossmatch_to_sample(sample, _synthetic_votes(), radius_arcsec=1.0)
    for col in _VOTE_COLUMNS:
        assert np.isnan(result[col].iloc[0])


def test_each_sample_row_gets_the_nearest_catalog_row():
    sample = pd.DataFrame({"ra": [10.0001, 20.0001], "dec": [0.0, 0.0]})
    result = crossmatch_to_sample(sample, _synthetic_votes(), radius_arcsec=1.0)
    assert result["iauname"].tolist() == ["a", "b"]
