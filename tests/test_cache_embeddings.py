"""_image_batch_loader (scripts/02_cache_embeddings.py) — band selection must be by name, not
list position, since legacy_south_all_images.parquet's `image_legacy` doesn't guarantee band
order. Imported via importlib since it lives in scripts/, not the installed package.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "02_cache_embeddings.py"
_spec = importlib.util.spec_from_file_location("cache_embeddings_script", _SCRIPT_PATH)
cache_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cache_script)


def _band(name: str, value: float, shape=(4, 4)) -> dict:
    return {
        "band": name,
        "flux": (np.ones(shape, dtype=np.float32) * value).tolist(),
        "mask": np.zeros(shape, dtype=bool).tolist(),
        "ivar": np.ones(shape, dtype=np.float32).tolist(),
        "psf_fwhm": 1.3,
        "scale": 0.262,
    }


def test_bands_selected_by_name_not_position():
    # Deliberately out of config order, plus an unrequested "i" band that must be ignored.
    pixels_by_id = {
        "obj1": [_band("z", 3.0), _band("g", 1.0), _band("r", 2.0), _band("i", 9.0)],
    }
    loader = cache_script._image_batch_loader(pixels_by_id, ["DES-G", "DES-R", "DES-Z"])
    batch = loader(["obj1"])
    pv = batch["pixel_values"]

    assert pv.shape == (1, 3, 4, 4)
    assert pv[0, 0, 0, 0].item() == 1.0  # G
    assert pv[0, 1, 0, 0].item() == 2.0  # R
    assert pv[0, 2, 0, 0].item() == 3.0  # Z


def test_missing_band_raises_clearly():
    pixels_by_id = {"obj1": [_band("g", 1.0), _band("r", 2.0)]}  # no z
    loader = cache_script._image_batch_loader(pixels_by_id, ["DES-G", "DES-R", "DES-Z"])
    with pytest.raises(KeyError, match="DES-Z"):
        loader(["obj1"])


def test_inconsistent_shapes_across_batch_raises():
    pixels_by_id = {
        "obj1": [_band("g", 1.0, (4, 4)), _band("r", 1.0, (4, 4)), _band("z", 1.0, (4, 4))],
        "obj2": [_band("g", 1.0, (8, 8)), _band("r", 1.0, (8, 8)), _band("z", 1.0, (8, 8))],
    }
    loader = cache_script._image_batch_loader(pixels_by_id, ["DES-G", "DES-R", "DES-Z"])
    with pytest.raises(ValueError, match="Inconsistent"):
        loader(["obj1", "obj2"])
