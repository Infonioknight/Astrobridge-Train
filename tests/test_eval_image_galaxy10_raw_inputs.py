"""eval/datasets/image_galaxy10.py's build_raw_inputs — regression coverage for a real bug caught
live on Modal: `image_bands` deserializes from parquet as triple-nested `dtype=object` numpy
arrays (outer (4,) object array -> each element a (96,) object array -> each element a real
(H,) float32 array), not one contiguous numeric block. A naive `np.asarray(b, dtype=np.float32)`
on the middle layer raises `ValueError: setting an array element with a sequence` — this test
builds a synthetic row with exactly that real nesting shape, not a flat/already-clean array,
so a regression back to the naive conversion fails here instead of only on a live Modal run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval.datasets.image_galaxy10 import build_raw_inputs


def _triple_nested_bands(n_bands: int = 4, h: int = 8, w: int = 8) -> np.ndarray:
    """Mirrors the real parquet-deserialized shape exactly: an object array of object arrays of
    real float32 arrays — not a plain nested Python list, and not a clean 3D numpy array.
    """
    outer = np.empty(n_bands, dtype=object)
    for b in range(n_bands):
        middle = np.empty(h, dtype=object)
        for row in range(h):
            middle[row] = np.arange(w, dtype=np.float32) + b * 100 + row
        outer[b] = middle
    return outer


def test_build_raw_inputs_handles_the_real_triple_nested_object_array_shape():
    row = pd.Series({"image_bands": _triple_nested_bands()})
    raw_inputs = build_raw_inputs(row)
    pixel_values = raw_inputs["image"]["pixel_values"]
    assert pixel_values.shape == (1, 3, 8, 8)
    assert pixel_values.dtype.is_floating_point


def test_build_raw_inputs_selects_grz_dropping_the_i_band():
    row = pd.Series({"image_bands": _triple_nested_bands()})
    raw_inputs = build_raw_inputs(row)
    pixel_values = raw_inputs["image"]["pixel_values"]
    # band values were built as b*100 + row + col, so band index is recoverable from the offset
    assert pixel_values[0, 0, 0, 0].item() == pytest.approx(0.0)      # g (band 0)
    assert pixel_values[0, 1, 0, 0].item() == pytest.approx(100.0)    # r (band 1)
    assert pixel_values[0, 2, 0, 0].item() == pytest.approx(300.0)    # z (band 3, i dropped)


def test_build_raw_inputs_rejects_wrong_band_count():
    row = pd.Series({"image_bands": _triple_nested_bands(n_bands=3)})
    with pytest.raises(ValueError, match="Expected 4 bands"):
        build_raw_inputs(row)
