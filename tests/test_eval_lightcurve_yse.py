"""eval/datasets/lightcurve_yse.py's raw_inputs assembly — pure logic (prepare_lightcurve_arrays
underneath is deliberately numpy-only, no torch, no network), exercised with synthetic arrays
matching the real column shapes/units confirmed against the live YSE dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from eval.datasets.lightcurve_yse import (
    SN_CLASS_CODES,
    SN_LABELS,
    build_raw_inputs_lightcurve,
    build_raw_inputs_with_image,
    render_lightcurve_plot,
    stratified_sample,
)
from eval.metrics.caption_to_label import predict_label_from_code


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


def test_sn_class_codes_covers_all_three_labels_in_order():
    assert SN_CLASS_CODES == {"0": "SN Ia", "1": "SN II", "2": "SN Ibc"}


def test_sn_class_codes_round_trips_through_predict_label_from_code():
    for code, label in SN_CLASS_CODES.items():
        assert predict_label_from_code(f" {code}", SN_CLASS_CODES) == label


def test_render_lightcurve_plot_returns_an_rgb_image():
    row = _lc_row()
    row["object_id"] = "obj1"
    image = render_lightcurve_plot(row)
    assert image.mode == "RGB"
    assert image.size[0] > 0 and image.size[1] > 0


def test_render_lightcurve_plot_handles_an_all_masked_object():
    # atcat_use all False (nothing accepted) must not crash — a real edge case, not hypothetical:
    # a badly-behaved object could plausibly have zero accepted points.
    row = _lc_row()
    row["object_id"] = "obj_empty"
    row["atcat_use"] = np.zeros(len(row["lc_mjd"]), dtype=bool)
    image = render_lightcurve_plot(row)
    assert image.mode == "RGB"


def _synthetic_sn_table() -> pd.DataFrame:
    labels, uids = [], []
    uid = 0
    for label, n in zip(SN_LABELS, [180, 71, 15]):  # real confirmed counts, see module docstring
        for _ in range(n):
            labels.append(label)
            uids.append(uid)
            uid += 1
    return pd.DataFrame({"class_label": labels, "uid": uids})


def test_stratified_sample_same_seed_is_reproducible():
    table = _synthetic_sn_table()
    s1 = stratified_sample(table, 90, seed=42, min_per_class=3)
    s2 = stratified_sample(table, 90, seed=42, min_per_class=3)
    assert s1["uid"].tolist() == s2["uid"].tolist()


def test_stratified_sample_covers_all_three_classes():
    table = _synthetic_sn_table()
    sample = stratified_sample(table, 90, seed=0, min_per_class=3)
    assert set(sample["class_label"]) == set(SN_LABELS)
    assert len(sample) == 90


def test_stratified_sample_rare_class_gets_at_least_min_per_class():
    table = _synthetic_sn_table()
    sample = stratified_sample(table, 90, seed=0, min_per_class=3)
    assert sample["class_label"].value_counts()["SN Ibc"] >= 3
