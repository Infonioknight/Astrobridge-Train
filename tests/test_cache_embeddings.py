"""_image_batch_loader (scripts/02_cache_embeddings.py) — band selection must be by canonical
name, not list position or an assumed exact string. Confirmed against a real run: the source
data's `band` field is a full string like "des-g" (lowercase, "des-" prefix), not a bare letter —
an earlier version of this test used bare "g"/"r"/"z" band strings, which matched an earlier
(buggy) implementation that only canonicalized the *configured* band label and not the *data's*
own band string, so "g" (canonicalized) never matched "des-g" (not canonicalized) in production
even though this test passed. Real bands observed: des-g, des-r, des-i, des-z.

Imported via importlib since it lives in scripts/, not the installed package.
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


def test_canonical_band_normalizes_every_observed_convention():
    # config-declared style, real-data style (confirmed), and a bare-letter fallback all -> "g"
    assert cache_script._canonical_band("DES-G") == "g"
    assert cache_script._canonical_band("des-g") == "g"
    assert cache_script._canonical_band("g") == "g"
    assert cache_script._canonical_band("G") == "g"
    assert cache_script._canonical_band("des_g") == "g"


def test_bands_selected_by_name_not_position_real_data_format():
    # Real format confirmed in production: full "des-<letter>" strings, deliberately out of
    # config order, plus an unrequested "des-i" band that must be ignored.
    pixels_by_id = {
        "obj1": [_band("des-z", 3.0), _band("des-g", 1.0), _band("des-r", 2.0), _band("des-i", 9.0)],
    }
    loader = cache_script._image_batch_loader(pixels_by_id, ["DES-G", "DES-R", "DES-Z"])
    batch = loader(["obj1"])
    pv = batch["pixel_values"]

    assert pv.shape == (1, 3, 4, 4)
    assert pv[0, 0, 0, 0].item() == 1.0  # G
    assert pv[0, 1, 0, 0].item() == 2.0  # R
    assert pv[0, 2, 0, 0].item() == 3.0  # Z


def test_bands_selected_by_name_bare_letter_format():
    # If some other object ends up with bare-letter band strings instead, that must work too.
    pixels_by_id = {
        "obj1": [_band("z", 3.0), _band("g", 1.0), _band("r", 2.0)],
    }
    loader = cache_script._image_batch_loader(pixels_by_id, ["DES-G", "DES-R", "DES-Z"])
    batch = loader(["obj1"])
    pv = batch["pixel_values"]

    assert pv[0, 0, 0, 0].item() == 1.0
    assert pv[0, 1, 0, 0].item() == 2.0
    assert pv[0, 2, 0, 0].item() == 3.0


def test_missing_band_raises_clearly():
    pixels_by_id = {"obj1": [_band("des-g", 1.0), _band("des-r", 2.0)]}  # no z
    loader = cache_script._image_batch_loader(pixels_by_id, ["DES-G", "DES-R", "DES-Z"])
    with pytest.raises(KeyError, match="DES-Z"):
        loader(["obj1"])


def test_inconsistent_shapes_across_batch_raises():
    pixels_by_id = {
        "obj1": [_band("des-g", 1.0, (4, 4)), _band("des-r", 1.0, (4, 4)), _band("des-z", 1.0, (4, 4))],
        "obj2": [_band("des-g", 1.0, (8, 8)), _band("des-r", 1.0, (8, 8)), _band("des-z", 1.0, (8, 8))],
    }
    loader = cache_script._image_batch_loader(pixels_by_id, ["DES-G", "DES-R", "DES-Z"])
    with pytest.raises(ValueError, match="Inconsistent"):
        loader(["obj1", "obj2"])


class TestFluxToArray:
    """Confirmed against a real run: np.asarray(flux, dtype=float32) can fail with 'setting an
    array element with a sequence' when a row near a survey/mosaic edge comes through as None
    instead of a properly-shaped row — see _flux_to_array's docstring for the full reasoning.
    """

    def test_clean_rectangular_flux_unaffected(self):
        arr = cache_script._flux_to_array([[1.0, 2.0], [3.0, 4.0]], "obj1", "g")
        assert arr.shape == (2, 2)
        assert arr.dtype == np.float32
        assert arr.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_none_rows_filled_with_nan(self):
        arr = cache_script._flux_to_array([[1.0, 2.0, 3.0], None, [4.0, 5.0, 6.0]], "obj2", "g")
        assert arr.shape == (3, 3)
        assert np.isnan(arr[1]).all()
        assert not np.isnan(arr[0]).any() and not np.isnan(arr[2]).any()

    def test_genuinely_different_row_widths_raises_not_silently_padded(self):
        # Real, non-None rows disagreeing on width must never be auto-padded/cropped — that
        # could silently distort the image — it must raise instead.
        with pytest.raises(ValueError, match="genuinely different"):
            cache_script._flux_to_array([[1.0, 2.0, 3.0], [4.0, 5.0]], "obj3", "g")

    def test_all_rows_none_raises_clearly(self):
        # numpy silently converts a top-level None to NaN even for a malformed 1D result — this
        # must be caught explicitly rather than let a wrongly-shaped array through unnoticed.
        with pytest.raises(ValueError, match="no usable rows at all"):
            cache_script._flux_to_array([None, None], "obj4", "g")
