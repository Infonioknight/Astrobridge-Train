"""eval/datasets/lightcurve_yse.py's raw_inputs assembly — pure logic (prepare_lightcurve_arrays
underneath is deliberately numpy-only, no torch, no network), exercised with synthetic arrays
matching the real column shapes/units confirmed against the live YSE dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from eval.datasets.lightcurve_yse import build_raw_inputs_lightcurve, build_raw_inputs_with_image


def _cfg():
    return OmegaConf.create({
        "modalities": {
            "lightcurve": {
                "max_tokens": 243,
                "encoder": {"kwargs": {"detection_window_days": 30, "detection_snr": 5.0, "subsample_seed": 0}},
            },
            "image": {"encoder": {"kwargs": {"bands": ["DES-G", "DES-R", "DES-Z"]}}},
        },
    })


def _lc_row(n: int = 10) -> pd.Series:
    return pd.Series({
        "object_id": "obj1",
        "lc_mjd": np.linspace(0, 100, n),
        "atcat_flux": np.full(n, 50.0),
        "atcat_flux_error": np.full(n, 1.0),
        "atcat_band_id": np.array([1, 2] * (n // 2)),
        "atcat_use": np.ones(n, dtype=bool),
    })


def test_build_raw_inputs_lightcurve_shape():
    raw_inputs = build_raw_inputs_lightcurve(_lc_row(), _cfg())
    assert set(raw_inputs) == {"lightcurve"}
    lc = raw_inputs["lightcurve"]
    assert set(lc) == {"flux", "flux_err", "time", "mask", "channel_index"}
    for arr in lc.values():
        assert arr.shape == (1, 243)  # batch dim added, padded to seq_len


def test_build_raw_inputs_with_image_checks_band_count():
    row_image = pd.Series({"object_id": "obj1", "image_flux": np.zeros((3, 8, 8), dtype=np.float32)})
    raw_inputs = build_raw_inputs_with_image(_lc_row(), row_image, _cfg())
    assert set(raw_inputs) == {"lightcurve", "image"}
    assert raw_inputs["image"]["pixel_values"].shape == (1, 3, 8, 8)


def test_build_raw_inputs_with_image_wrong_band_count_raises():
    row_image = pd.Series({"object_id": "obj1", "image_flux": np.zeros((4, 8, 8), dtype=np.float32)})
    with pytest.raises(ValueError, match="expected 3"):
        build_raw_inputs_with_image(_lc_row(), row_image, _cfg())
